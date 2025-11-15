import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    wandb.init(
        project="nlp-hw4-text2sql",
        name=args.experiment_name,
        config=vars(args)
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.

    WHAT: Load T5-small model either with pretrained weights or from scratch
    WHY:
      - Pretrained weights give us a huge head start (trained on C4 dataset)
      - T5-small is the right balance: 60M params, fits in most GPUs
      - From scratch is harder but good for understanding/extra credit
    '''
    if args.finetune:
        # BASELINE APPROACH: Load pretrained model
        # This is what you want for good baseline results!
        print("Loading pretrained T5-small model...")
        model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
    else:
        # ALTERNATIVE: Train from scratch (Extra Credit)
        print("Initializing T5-small from scratch (random weights)...")
        config = T5Config.from_pretrained('google-t5/t5-small')
        model = T5ForConditionalGeneration(config)

    # Move model to GPU if available (critical for speed!)
    model = model.to(DEVICE)

    # Print model stats for debugging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized on {DEVICE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    '''
    Save model checkpoint to disk.

    WHAT: Save model weights to a .pt file
    WHY:
      - We save 'best' model (highest F1) and 'last' model (most recent)
      - Allows us to resume training or use best model for final evaluation
      - Only save state_dict (weights), not entire model, for efficiency
    '''
    mkdir(checkpoint_dir)

    if best:
        save_path = os.path.join(checkpoint_dir, 'best_model.pt')
        print(f"Saving BEST model (highest dev F1 so far)...")
    else:
        save_path = os.path.join(checkpoint_dir, 'last_model.pt')

    # Save model state dict (just the weights, not the architecture)
    # WHY: More portable and takes less space than saving entire model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, save_path)

    if best:
        print(f"✓ Best model saved to {save_path}")

def load_model_from_checkpoint(args, best):
    '''
    Load a previously saved model checkpoint.

    WHAT: Reconstruct model and load saved weights
    WHY: Need to use best model for final test evaluation
    '''
    if best:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        print(f"Loading BEST model from {checkpoint_path}")
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, 'last_model.pt')
        print(f"Loading LAST model from {checkpoint_path}")

    # First recreate the model architecture
    model = initialize_model(args)

    # Then load the saved weights
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Model loaded successfully")
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, transformers.pytorch_utils.ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        pass

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

