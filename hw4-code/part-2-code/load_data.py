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
        # ========== STEP 1: Load raw text files ==========
        print(f"Loading data from {data_folder}/{split}...")

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
        print("Tokenizing natural language queries...")
        self.encoder_inputs = []

        for nl in tqdm(self.nl_queries, desc="Encoding inputs"):
            # Add task prefix - T5 was pretrained with task descriptions!
            # WHY: Helps model understand what transformation to perform
            text = f"translate English to SQL: {nl}"

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