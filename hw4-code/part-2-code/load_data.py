import sys
import shutil
import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        WHAT: Initialize the dataset by loading .nl and .sql files and tokenizing them
        WHY:
          - We need to convert text to token IDs that T5 can process
          - Different behavior for train/dev (have labels) vs test (no labels)

        BASELINE STRATEGY:
          - Use T5's pretrained tokenizer (handles both NL and SQL well enough)
          - Add task prefix to help model understand the task
          - Process everything upfront (not on-the-fly) for speed
        '''
        print(f"\n{'='*60}")
        print(f"Initializing T5Dataset for {split} split")
        print(f"{'='*60}")

        # Initialize T5 tokenizer from pretrained checkpoint
        # WHY: T5 tokenizer uses SentencePiece, works well for both NL and code
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split

        # Load and process the data
        self.process_data(data_folder, split, self.tokenizer)

        print(f"✓ Dataset initialized with {len(self)} examples")

    def process_data(self, data_folder, split, tokenizer):
        '''
        WHAT: Load files and tokenize both inputs and outputs
        WHY: Need to prepare data in the format T5 expects
        '''
        # ========== STEP 1: Load schema and raw text files ==========
        print(f"Loading data from {data_folder}/{split}...")

        # Load database schema
        schema_path = os.path.join(data_folder, 'schema_prompt.txt')
        with open(schema_path, 'r') as f:
            self.schema_text = f.read().strip()
        print(f"  - Loaded database schema")

        nl_path = os.path.join(data_folder, f'{split}.nl')
        with open(nl_path, 'r') as f:
            self.nl_queries = [line.strip() for line in f.readlines()]

        print(f"  - Loaded {len(self.nl_queries)} natural language queries")

        # Load SQL only for train/dev (test doesn't have labels)
        if split != 'test':
            sql_path = os.path.join(data_folder, f'{split}.sql')
            with open(sql_path, 'r') as f:
                self.sql_queries = [line.strip() for line in f.readlines()]
            print(f"  - Loaded {len(self.sql_queries)} SQL queries")
        else:
            self.sql_queries = None
            print(f"  - No SQL queries (test set)")

        # ========== STEP 2: Tokenize encoder inputs (NL queries) ==========
        print("Tokenizing natural language queries with schema context...")
        self.encoder_inputs = []

        for nl in tqdm(self.nl_queries, desc="Encoding inputs"):
            # Include database schema in the prompt (blog-style improvement!)
            # WHY: Provides table/column context so model doesn't need to memorize schema
            text = f"""{self.schema_text}

Question: {nl}

SQL:"""

            # Tokenize with T5 tokenizer
            # add_special_tokens=True adds </s> (EOS) automatically
            tokens = tokenizer(text, return_tensors='pt', add_special_tokens=True)
            self.encoder_inputs.append(tokens['input_ids'].squeeze(0))

        # ========== STEP 3: Tokenize decoder inputs/targets (SQL) ==========
        if self.sql_queries is not None:
            print("Tokenizing SQL queries...")
            self.decoder_inputs = []
            self.decoder_targets = []

            for sql in tqdm(self.sql_queries, desc="Encoding outputs"):
                # Tokenize SQL query
                tokens = tokenizer(sql, return_tensors='pt', add_special_tokens=True)
                token_ids = tokens['input_ids'].squeeze(0)

                # DECODER INPUT: We need to shift by adding a start token
                # T5 uses pad_token_id (0) as decoder start token
                # Format: [0, token1, token2, ..., tokenN-1]
                # WHY: During training, decoder sees previous tokens (teacher forcing)
                start_token = tokenizer.pad_token_id
                decoder_input = torch.cat([
                    torch.tensor([start_token]),
                    token_ids[:-1]  # Remove EOS, we'll predict it
                ])
                self.decoder_inputs.append(decoder_input)

                # DECODER TARGET: The tokens we want to predict
                # Format: [token1, token2, ..., tokenN, EOS]
                # WHY: We train the model to predict the next token at each step
                self.decoder_targets.append(token_ids)
        else:
            # For test set, just need the initial decoder token for generation
            self.initial_decoder_token = torch.tensor([tokenizer.pad_token_id])

    def __len__(self):
        '''Return the number of examples in the dataset'''
        return len(self.nl_queries)

    def __getitem__(self, idx):
        '''
        WHAT: Return a single training example
        WHY: DataLoader calls this to build batches
        '''
        if self.split != 'test':
            # For train/dev: return input, decoder input, and target
            return (
                self.encoder_inputs[idx],      # NL query tokens
                self.decoder_inputs[idx],      # SQL tokens shifted right
                self.decoder_targets[idx]      # SQL tokens (what we predict)
            )
        else:
            # For test: only return input and initial token
            return (
                self.encoder_inputs[idx],
                self.initial_decoder_token
            )

def normal_collate_fn(batch):
    '''
    WHAT: Combine multiple examples into a single batch with padding
    WHY:
      - Different examples have different lengths
      - We need to pad them to the same length for parallel processing
      - Dynamic padding = only pad to longest in THIS batch (efficient!)

    BASELINE APPROACH:
      - Use PyTorch's pad_sequence for automatic padding
      - Create attention mask to ignore padding tokens
      - Pad with 0 (PAD_IDX) which T5 expects

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Each item in batch is a tuple: (encoder_input, decoder_input, decoder_target)
    # Unzip them into separate lists
    encoder_inputs = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]

    # ========== PAD SEQUENCES ==========
    # pad_sequence: pads to length of longest sequence in batch
    # batch_first=True: output shape is [batch_size, max_length]
    # padding_value=PAD_IDX: use 0 for padding (T5 standard)
    # WHY: GPU operations require fixed-size tensors
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)

    # ========== CREATE ATTENTION MASK ==========
    # Attention mask: 1 for real tokens, 0 for padding
    # WHY: Tells model which tokens to pay attention to
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # ========== EXTRACT INITIAL DECODER TOKEN ==========
    # For generation during evaluation, we need just the first token
    # Shape: [batch_size, 1]
    initial_decoder_inputs = decoder_input_ids[:, 0:1]

    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    WHAT: Collate function for test set (no labels available)
    WHY:
      - Test set doesn't have SQL targets, so simpler than normal_collate_fn
      - Only need encoder inputs and initial decoder token for generation

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Each item in batch is a tuple: (encoder_input, initial_decoder_token)
    encoder_inputs = [item[0] for item in batch]
    initial_tokens = [item[1] for item in batch]

    # Pad encoder inputs to same length
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)

    # Create attention mask (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Stack initial decoder tokens into a batch
    # Shape: [batch_size, 1]
    initial_decoder_inputs = torch.stack(initial_tokens)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING load_data.py FUNCTIONS")
    print("="*80)

    # Create output directory for analysis files
    output_dir = "load_data_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCreated output directory: {output_dir}/")

    # Test parameters
    batch_size = 4
    test_batch_size = 8

    with open(f"{output_dir}/analysis.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LOAD_DATA.PY ANALYSIS OUTPUT\n")
        f.write("="*80 + "\n\n")

        # ========== Test 1: Load dataloaders ==========
        print("\n" + "-"*80)
        print("TEST 1: Loading dataloaders")
        print("-"*80)
        f.write("\n" + "-"*80 + "\n")
        f.write("TEST 1: LOADING DATALOADERS\n")
        f.write("-"*80 + "\n")

        try:
            train_loader, dev_loader, test_loader = load_t5_data(batch_size, test_batch_size)

            f.write(f"\n✓ Successfully loaded all dataloaders\n")
            f.write(f"  - Train batches: {len(train_loader)}\n")
            f.write(f"  - Dev batches: {len(dev_loader)}\n")
            f.write(f"  - Test batches: {len(test_loader)}\n")
            f.write(f"  - Train batch size: {batch_size}\n")
            f.write(f"  - Test/Dev batch size: {test_batch_size}\n")

            print(f"✓ Train loader: {len(train_loader)} batches")
            print(f"✓ Dev loader: {len(dev_loader)} batches")
            print(f"✓ Test loader: {len(test_loader)} batches")

        except Exception as e:
            error_msg = f"✗ Error loading dataloaders: {str(e)}\n"
            f.write(error_msg)
            print(error_msg)
            raise

        # ========== Test 2: Examine train batch ==========
        print("\n" + "-"*80)
        print("TEST 2: Examining train batch structure")
        print("-"*80)
        f.write("\n" + "-"*80 + "\n")
        f.write("TEST 2: TRAIN BATCH STRUCTURE\n")
        f.write("-"*80 + "\n\n")

        train_batch = next(iter(train_loader))
        encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs = train_batch

        f.write(f"Train batch components:\n")
        f.write(f"  1. encoder_ids shape: {encoder_ids.shape}\n")
        f.write(f"  2. encoder_mask shape: {encoder_mask.shape}\n")
        f.write(f"  3. decoder_input_ids shape: {decoder_input_ids.shape}\n")
        f.write(f"  4. decoder_target_ids shape: {decoder_target_ids.shape}\n")
        f.write(f"  5. initial_decoder_inputs shape: {initial_decoder_inputs.shape}\n\n")

        print(f"Encoder IDs shape: {encoder_ids.shape}")
        print(f"Encoder mask shape: {encoder_mask.shape}")
        print(f"Decoder input IDs shape: {decoder_input_ids.shape}")
        print(f"Decoder target IDs shape: {decoder_target_ids.shape}")
        print(f"Initial decoder inputs shape: {initial_decoder_inputs.shape}")

        # ========== Test 3: Decode and save examples ==========
        print("\n" + "-"*80)
        print("TEST 3: Decoding sample examples")
        print("-"*80)
        f.write("-"*80 + "\n")
        f.write("TEST 3: DECODED SAMPLE EXAMPLES (First batch)\n")
        f.write("-"*80 + "\n\n")

        tokenizer = train_loader.dataset.tokenizer

        for i in range(min(batch_size, encoder_ids.shape[0])):
            f.write(f"\nExample {i+1}:\n")
            f.write("-" * 40 + "\n")

            # Decode encoder input (NL query)
            encoder_text = tokenizer.decode(encoder_ids[i], skip_special_tokens=False)
            f.write(f"ENCODER INPUT (NL):\n{encoder_text}\n\n")

            # Decode decoder input
            decoder_input_text = tokenizer.decode(decoder_input_ids[i], skip_special_tokens=False)
            f.write(f"DECODER INPUT:\n{decoder_input_text}\n\n")

            # Decode decoder target (SQL query)
            decoder_target_text = tokenizer.decode(decoder_target_ids[i], skip_special_tokens=False)
            f.write(f"DECODER TARGET (SQL):\n{decoder_target_text}\n\n")

            # Show mask
            mask_str = "".join(["1" if m else "0" for m in encoder_mask[i][:20]])
            f.write(f"ENCODER MASK (first 20): {mask_str}...\n")
            f.write(f"INITIAL DECODER TOKEN: {initial_decoder_inputs[i].tolist()}\n")

            print(f"\nExample {i+1} - NL: {encoder_text[:60]}...")
            print(f"Example {i+1} - SQL: {decoder_target_text[:60]}...")

        # ========== Test 4: Test set batch ==========
        print("\n" + "-"*80)
        print("TEST 4: Examining test batch structure")
        print("-"*80)
        f.write("\n" + "-"*80 + "\n")
        f.write("TEST 4: TEST BATCH STRUCTURE\n")
        f.write("-"*80 + "\n\n")

        test_batch = next(iter(test_loader))
        test_encoder_ids, test_encoder_mask, test_initial_decoder = test_batch

        f.write(f"Test batch components (no labels):\n")
        f.write(f"  1. encoder_ids shape: {test_encoder_ids.shape}\n")
        f.write(f"  2. encoder_mask shape: {test_encoder_mask.shape}\n")
        f.write(f"  3. initial_decoder_inputs shape: {test_initial_decoder.shape}\n\n")

        print(f"Test encoder IDs shape: {test_encoder_ids.shape}")
        print(f"Test encoder mask shape: {test_encoder_mask.shape}")
        print(f"Test initial decoder shape: {test_initial_decoder.shape}")

        # Decode first test example
        f.write("First test example:\n")
        test_text = tokenizer.decode(test_encoder_ids[0], skip_special_tokens=False)
        f.write(f"ENCODER INPUT (NL): {test_text}\n")
        f.write(f"INITIAL DECODER TOKEN: {test_initial_decoder[0].tolist()}\n\n")

        print(f"First test example: {test_text[:60]}...")

        # ========== Test 5: Dataset statistics ==========
        print("\n" + "-"*80)
        print("TEST 5: Dataset statistics")
        print("-"*80)
        f.write("-"*80 + "\n")
        f.write("TEST 5: DATASET STATISTICS\n")
        f.write("-"*80 + "\n\n")

        train_dataset = train_loader.dataset
        dev_dataset = dev_loader.dataset
        test_dataset = test_loader.dataset

        # Calculate sequence length statistics
        train_encoder_lengths = [len(seq) for seq in train_dataset.encoder_inputs]
        train_decoder_lengths = [len(seq) for seq in train_dataset.decoder_targets]

        f.write(f"TRAIN DATASET:\n")
        f.write(f"  Total examples: {len(train_dataset)}\n")
        f.write(f"  Encoder lengths - Min: {min(train_encoder_lengths)}, Max: {max(train_encoder_lengths)}, Avg: {sum(train_encoder_lengths)/len(train_encoder_lengths):.2f}\n")
        f.write(f"  Decoder lengths - Min: {min(train_decoder_lengths)}, Max: {max(train_decoder_lengths)}, Avg: {sum(train_decoder_lengths)/len(train_decoder_lengths):.2f}\n\n")

        dev_encoder_lengths = [len(seq) for seq in dev_dataset.encoder_inputs]
        dev_decoder_lengths = [len(seq) for seq in dev_dataset.decoder_targets]

        f.write(f"DEV DATASET:\n")
        f.write(f"  Total examples: {len(dev_dataset)}\n")
        f.write(f"  Encoder lengths - Min: {min(dev_encoder_lengths)}, Max: {max(dev_encoder_lengths)}, Avg: {sum(dev_encoder_lengths)/len(dev_encoder_lengths):.2f}\n")
        f.write(f"  Decoder lengths - Min: {min(dev_decoder_lengths)}, Max: {max(dev_decoder_lengths)}, Avg: {sum(dev_decoder_lengths)/len(dev_decoder_lengths):.2f}\n\n")

        test_encoder_lengths = [len(seq) for seq in test_dataset.encoder_inputs]

        f.write(f"TEST DATASET:\n")
        f.write(f"  Total examples: {len(test_dataset)}\n")
        f.write(f"  Encoder lengths - Min: {min(test_encoder_lengths)}, Max: {max(test_encoder_lengths)}, Avg: {sum(test_encoder_lengths)/len(test_encoder_lengths):.2f}\n")
        f.write(f"  No decoder targets (test set)\n\n")

        print(f"\nTrain dataset: {len(train_dataset)} examples")
        print(f"Dev dataset: {len(dev_dataset)} examples")
        print(f"Test dataset: {len(test_dataset)} examples")

        # ========== Test 6: Save sample raw data ==========
        print("\n" + "-"*80)
        print("TEST 6: Saving raw data samples")
        print("-"*80)
        f.write("-"*80 + "\n")
        f.write("TEST 6: RAW DATA SAMPLES\n")
        f.write("-"*80 + "\n\n")

        with open(f"{output_dir}/train_samples.txt", "w") as train_file:
            train_file.write("TRAIN SET SAMPLES (First 10)\n")
            train_file.write("="*80 + "\n\n")
            for i in range(min(10, len(train_dataset.nl_queries))):
                train_file.write(f"Example {i+1}:\n")
                train_file.write(f"NL:  {train_dataset.nl_queries[i]}\n")
                train_file.write(f"SQL: {train_dataset.sql_queries[i]}\n\n")

        with open(f"{output_dir}/dev_samples.txt", "w") as dev_file:
            dev_file.write("DEV SET SAMPLES (First 10)\n")
            dev_file.write("="*80 + "\n\n")
            for i in range(min(10, len(dev_dataset.nl_queries))):
                dev_file.write(f"Example {i+1}:\n")
                dev_file.write(f"NL:  {dev_dataset.nl_queries[i]}\n")
                dev_file.write(f"SQL: {dev_dataset.sql_queries[i]}\n\n")

        with open(f"{output_dir}/test_samples.txt", "w") as test_file:
            test_file.write("TEST SET SAMPLES (First 10)\n")
            test_file.write("="*80 + "\n\n")
            for i in range(min(10, len(test_dataset.nl_queries))):
                test_file.write(f"Example {i+1}:\n")
                test_file.write(f"NL:  {test_dataset.nl_queries[i]}\n")
                test_file.write(f"SQL: (no labels for test set)\n\n")

        f.write("✓ Saved raw data samples to separate files\n")
        print("✓ Saved train_samples.txt")
        print("✓ Saved dev_samples.txt")
        print("✓ Saved test_samples.txt")

        # ========== Summary ==========
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY\n")
        f.write("="*80 + "\n")
        f.write("All tests completed successfully!\n")
        f.write(f"\nOutput files created in {output_dir}/:\n")
        f.write("  1. analysis.txt - This file with all test results\n")
        f.write("  2. train_samples.txt - First 10 training examples\n")
        f.write("  3. dev_samples.txt - First 10 dev examples\n")
        f.write("  4. test_samples.txt - First 10 test examples\n")

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED!")
    print("="*80)
    print(f"\nOutput files saved in '{output_dir}/' directory:")
    print(f"  - analysis.txt")
    print(f"  - train_samples.txt")
    print(f"  - dev_samples.txt")
    print(f"  - test_samples.txt")
    print("\nYou can now analyze these files to verify everything works as expected.")
    print("="*80)