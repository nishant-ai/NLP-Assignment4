import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    words = word_tokenize(text)

    # Target: modify approximately 15% of words with typos
    typo_probability = 0.15

    transformed_words = []
    for word in words:
        # Only apply typos to words longer than 3 characters
        # Skip punctuation and very short words
        if len(word) > 3 and word.isalpha() and random.random() < typo_probability:
            # Randomly choose a typo type
            typo_type = random.choice(['swap', 'delete', 'duplicate'])

            if typo_type == 'swap' and len(word) > 4:
                # Swap two adjacent characters (not first or last)
                pos = random.randint(1, len(word) - 3)
                word_list = list(word)
                word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
                word = ''.join(word_list)

            elif typo_type == 'delete':
                # Delete a random character (not first or last)
                pos = random.randint(1, len(word) - 2)
                word = word[:pos] + word[pos + 1:]

            elif typo_type == 'duplicate':
                # Duplicate a random character
                pos = random.randint(1, len(word) - 2)
                word = word[:pos] + word[pos] + word[pos:]

        transformed_words.append(word)

    # Detokenize back to sentence
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_words)

    ##### YOUR CODE ENDS HERE ######

    return example
