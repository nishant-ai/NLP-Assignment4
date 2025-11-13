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
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.encoder_inputs, self.decoder_targets = self.process_data(data_folder,split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        """
        Load .nl and .sql files, tokenize them, and return tokenized data
        """
        # 1. Build File Paths
        nl_file = os.path.join(data_folder, f'{split}.nl')
        sql_file = os.path.join(data_folder, f'{split}.sql')

        # 2. Read the files
        nl_queries = load_lines(nl_file) # Natural language queries
        
        if split == "test":
            sql_queries = ['' for _ in nl_queries] # Empty Placeholders
        else:
            sql_queries = load_lines(sql_file) # SQL queries

        # 3. Tokenize encoder inputs (NL)
        encoder_inputs = []

        for nl_query in nl_queries:
            encoded = tokenizer(nl_query,
                                truncation=True,
                                max_length=512,
                                return_tensors='pt',
                                padding=False # We'll do dynamic padding in collate_fn
                                )

            encoder_inputs.append(encoded['input_ids'].squeeze(0))

        # 4. Tokenize decoder targets (SQL)
        decoder_targets = []

        for sql_query in sql_queries:
            if sql_query:
                encoded = tokenizer(
                    sql_query,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                    padding=False
                )
                decoder_targets.append(encoded['input_ids'].squeeze(0))
            else:
                decoder_targets.append(torch.tensor([])) # Empty for test set

        return encoder_inputs, decoder_targets

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        encoder_input = self.encoder_inputs[idx]
        decoder_target = self.decoder_targets[idx]
        
        return encoder_input, decoder_target

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

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
    # 1. Separate encoder inputs and decoder targets
    encoder_inputs = [item[0] for item in batch]
    decoder_targets = [item[1] for item in batch]
    
    # 2. Pad encoder inputs to same length
    # pad_sequence adds padding (PAD_IDX=0) to make all sequences same length
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # 3. Create attention mask for encoder (1 = real token, 0 = padding)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # 4. Create decoder inputs (shift right + add BOS token)
    # T5 uses <extra_id_0> as BOS token (token_id = 32099)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.additional_special_tokens_ids[0]  # <extra_id_0>
    
    # Prepend BOS token to each decoder target
    decoder_inputs = []
    for target in decoder_targets:
        # Create decoder input: [BOS, target[0], target[1], ..., target[n-1]]
        decoder_input = torch.cat([torch.tensor([bos_token_id]), target[:-1]])
        decoder_inputs.append(decoder_input)
    
    # 5. Pad decoder inputs
    decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # 6. Pad decoder targets (what we want to predict)
    decoder_target_ids = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # 7. Initial decoder inputs (for generation during evaluation)
    initial_decoder_inputs = torch.full((len(batch), 1), bos_token_id, dtype=torch.long)
    
    return encoder_ids, encoder_mask, decoder_input_ids, decoder_target_ids, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # 1. Get encoder inputs only
    encoder_inputs = [item[0] for item in batch]
    
    # 2. Pad encoder inputs
    encoder_ids = pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_IDX)
    
    # 3. Create attention mask
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    # 4. Create initial decoder input (BOS token)
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_token_id = tokenizer.additional_special_tokens_ids[0]
    initial_decoder_inputs = torch.full((len(batch), 1), bos_token_id, dtype=torch.long)
    
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
    """
    Load data for prompting experiments (used with Gemma models)
    Returns raw text strings instead of tokenized tensors

    Returns:
        train_x: List of training natural language queries
        train_y: List of training SQL queries
        dev_x: List of dev natural language queries
        dev_y: List of dev SQL queries
        test_x: List of test natural language queries (no ground truth SQL)
    """
    # Training data
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))

    # Dev data
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))

    # Test data (no ground truth SQL)
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))

    return train_x, train_y, dev_x, dev_y, test_x

# Custom Function
def compute_data_statistics():
    """Compute statistics for Q4"""
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    for split in ['train', 'dev']:
        nl_file = f'data/{split}.nl'
        sql_file = f'data/{split}.sql'
        
        nl_queries = load_lines(nl_file)
        sql_queries = load_lines(sql_file)
        
        # Number of examples
        num_examples = len(nl_queries)
        
        # Tokenize all queries
        nl_tokens = [tokenizer(q)['input_ids'] for q in nl_queries]
        sql_tokens = [tokenizer(q)['input_ids'] for q in sql_queries]
        
        # Mean lengths
        mean_nl_length = np.mean([len(t) for t in nl_tokens])
        mean_sql_length = np.mean([len(t) for t in sql_tokens])
        
        # Vocabulary sizes
        nl_vocab = set()
        for tokens in nl_tokens:
            nl_vocab.update(tokens)
        
        sql_vocab = set()
        for tokens in sql_tokens:
            sql_vocab.update(tokens)
        
        print(f"\n=== {split.upper()} SET ===")
        print(f"Number of examples: {num_examples}")
        print(f"Mean NL length: {mean_nl_length:.2f}")
        print(f"Mean SQL length: {mean_sql_length:.2f}")
        print(f"NL vocab size: {len(nl_vocab)}")
        print(f"SQL vocab size: {len(sql_vocab)}")

if __name__ == "__main__":
    # Test the dataset
    dataset = T5Dataset('data', 'train')
    print(f"Dataset size: {len(dataset)}")
    
    # Get one example
    enc, dec = dataset[0]
    print(f"\nEncoder input shape: {enc.shape}")
    print(f"Decoder target shape: {dec.shape}")
    
    # Decode to see actual text
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    print(f"\nNatural language: {tokenizer.decode(enc)}")
    print(f"SQL query: {tokenizer.decode(dec)}")
    
    # Test dataloader
    train_loader = get_dataloader(batch_size=4, split='train')
    for batch in train_loader:
        enc_ids, enc_mask, dec_in, dec_tgt, initial = batch
        print(f"\n--- Batch shapes ---")
        print(f"Encoder IDs: {enc_ids.shape}")
        print(f"Encoder mask: {enc_mask.shape}")
        print(f"Decoder inputs: {dec_in.shape}")
        print(f"Decoder targets: {dec_tgt.shape}")
        print(f"Initial decoder: {initial.shape}")
        break