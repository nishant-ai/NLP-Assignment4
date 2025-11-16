import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir
    experiment_name = 'ft_experiment'
    
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    
    for epoch in range(args.max_n_epochs):
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad ])
        
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    WHAT: Evaluate model on dev set during training
    WHY:
      - Need to track progress and decide when to stop training
      - Compute both loss (for training signal) and F1 (for actual performance)
      - Save best model based on F1, not loss

    BASELINE APPROACH:
      - Use beam search (num_beams=5) for better quality than greedy
      - Compute loss to track training progress
      - Execute generated SQL on database to get F1 score
      - Track error rate to identify syntax issues
    '''
    from transformers import T5TokenizerFast

    print("\n" + "="*60)
    print("EVALUATING ON DEV SET")
    print("="*60)

    model.eval()  # Set to eval mode (disables dropout, etc.)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    # For loss calculation
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    # For SQL generation
    all_generated_queries = []

    with torch.no_grad():  # Don't compute gradients during evaluation
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_input in tqdm(dev_loader, desc="Evaluating"):
            # Move to GPU
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)

            # ========== STEP 1: Compute loss (for monitoring) ==========
            # Forward pass with teacher forcing
            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )['logits']

            # Compute loss only on non-padding tokens
            # WHY: Padding tokens are meaningless, shouldn't affect loss
            non_pad = decoder_targets != PAD_IDX
            loss = criterion(logits[non_pad], decoder_targets[non_pad])

            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # ========== STEP 2: Generate SQL queries ==========
            # Use beam search for better quality outputs
            # WHY: Beam search explores multiple hypotheses, better than greedy
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=256,      # Max new tokens to generate (not including input)
                num_beams=5,             # BASELINE: 5 beams is good balance
                early_stopping=True,     # Stop when all beams have EOS
                decoder_start_token_id=tokenizer.pad_token_id,  # Match training setup
                repetition_penalty=1.2,  # Penalize repeated tokens/phrases
                no_repeat_ngram_size=3,  # Don't repeat any 3-gram
            )

            # Decode token IDs back to text
            # skip_special_tokens=True removes <pad>, </s>, etc.
            generated_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_queries.extend(generated_queries)

    # ========== STEP 3: Save queries and execute on database ==========
    print(f"\nGenerated {len(all_generated_queries)} SQL queries")
    print("Executing queries on database...")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)

    # ========== STEP 4: Compute metrics ==========
    print("Computing metrics...")
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_pth, model_sql_path,
        gt_record_path, model_record_path
    )

    # Calculate error rate (how many queries caused SQL errors)
    num_errors = sum(1 for msg in error_msgs if msg != "")
    error_rate = num_errors / len(error_msgs)

    # Average loss
    avg_loss = total_loss / total_tokens

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Record F1: {record_f1:.4f} (PRIMARY METRIC)")
    print(f"  Record EM: {record_em:.4f}")
    print(f"  SQL EM: {sql_em:.4f}")
    print(f"  Error Rate: {error_rate*100:.2f}%")
    print(f"{'='*60}\n")

    return avg_loss, record_f1, record_em, sql_em, error_rate
        
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    WHAT: Run inference on test set and save predictions
    WHY:
      - Generate final predictions for submission
      - No ground truth available, so can't compute metrics
      - Save SQL queries and database records for grading

    IMPLEMENTATION: Almost identical to eval_epoch, but:
      - No loss computation (no targets)
      - No metrics computation (no ground truth)
      - Just generate and save
    '''
    from transformers import T5TokenizerFast

    print("\n" + "="*60)
    print("RUNNING INFERENCE ON TEST SET")
    print("="*60)

    model.eval()  # Set to eval mode
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

    all_generated_queries = []

    with torch.no_grad():  # No gradients needed for inference
        for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader, desc="Generating SQL"):
            # Move to GPU
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)

            # ========== GENERATE SQL QUERIES ==========
            # Same generation settings as evaluation
            # WHY: Consistency with dev set evaluation
            generated_ids = model.generate(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                max_new_tokens=256,      # Same as dev
                num_beams=5,             # Same as dev
                early_stopping=True,
                decoder_start_token_id=tokenizer.pad_token_id,  # Match training setup
                repetition_penalty=1.2,  # Penalize repeated tokens/phrases
                no_repeat_ngram_size=3,  # Don't repeat any 3-gram
            )

            # Decode to text
            generated_queries = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_generated_queries.extend(generated_queries)

    # ========== SAVE RESULTS ==========
    print(f"\nGenerated {len(all_generated_queries)} SQL queries for test set")
    print("Executing queries on database and saving results...")
    save_queries_and_records(all_generated_queries, model_sql_path, model_record_path)

    print(f"\n{'='*60}")
    print(f"TEST INFERENCE COMPLETE!")
    print(f"  SQL queries saved to: {model_sql_path}")
    print(f"  Database records saved to: {model_record_path}")
    print(f"{'='*60}\n")
    print("IMPORTANT: Submit these files to Gradescope for grading!")

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = 'ft_experiment'
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_em, dev_record_f1, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print("Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

if __name__ == "__main__":
    main()
