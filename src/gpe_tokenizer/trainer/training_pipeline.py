import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
import numpy as np
from collections import Counter
import logging
import itertools


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SinhalaGPETokenizerTrainer:

    def __init__(self, dataset_size, vocab_size, filepath=None, dataset=None, output_dir="src/gpe_tokenizer/models"):
        self.DUMMY_PREFIX = " "
        self.DATASET_SIZE = dataset_size
        self.VOCAB_SIZE = vocab_size
        self.start_time = time.time()
        self.vocab = {}
        self.vocab_re = {}
        self.merges = {}
        self.output_dir = output_dir

        if filepath and dataset:
            raise ValueError("Provide either filepath or dataset, not both.")
        
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        elif dataset:
            self.lines = dataset

        self.lines = self.lines[:self.DATASET_SIZE]

    def calculate_elapsed_time(self):
        end_time = time.time()
        td = end_time - self.start_time
        days, remainder = divmod(td, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return int(days), int(hours), int(minutes), int(seconds)

    def save_pickle(self, dictionary, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(dictionary, file)

    def save_models(self):
        self.save_pickle(self.vocab, os.path.join(self.output_dir, "vocab.pkl"))
        self.save_pickle(self.vocab_re, os.path.join(self.output_dir, "vocab_re.pkl"))
        self.save_pickle(self.merges, os.path.join(self.output_dir, "merges.pkl"))


    def merge(self, ids, pair, idx):
        # Early exit if pair not in ids
        if pair[0] not in ids or pair[1] not in ids:
            return ids

        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    

    def get_stats(self, ids_list):
        counts = Counter()
        for ids in ids_list:
            counts.update(zip(ids, ids[1:]))
        return counts

    # ------------------------
    # Grapheme tokenization with caching
    # ------------------------
    def tokenize_graphemes(self, text):
        """Return list of graphemes for a word."""
        return list(grapheme.graphemes(text))

    # ------------------------
    # Build initial vocab
    # ------------------------
    def build_vocab(self):
        # Flatten all graphemes in corpus
        all_graphemes = itertools.chain.from_iterable(
            self.tokenize_graphemes(word)
            for line in tqdm(self.lines, desc="Building initial vocab")
            for word in line.split()
        )
        
        # Count frequency (optional, if you want sorted vocab)
        counts = Counter(all_graphemes)
        
        self.graphemes_list = list(counts.keys())
        self.vocab = {i: g for i, g in enumerate(self.graphemes_list)}
        self.vocab_re = {g: i for i, g in enumerate(self.graphemes_list)}
        
        logger.info(f"Initial vocab size: {len(self.vocab)}")

    # ------------------------
    # Convert text to ID sequences
    # ------------------------
    def convert_to_ids(self):
        self.ids_list = []
        for line in tqdm(self.lines, desc="Converting text to IDs"):
            words = line.split()
            for word in words:
                self.ids_list.append([self.vocab_re[g] for g in self.tokenize_graphemes(word)])



    # ------------------------
    # Train BPE
    # ------------------------
    def train(self):
        logger.info("Starting training...")
        self.build_vocab()
        self.convert_to_ids()

        self.merges = {}

        for i in tqdm(range(self.VOCAB_SIZE), desc="Training BPE"):
            stats = self.get_stats(self.ids_list)
            if not stats:
                print("No more pairs to merge!")
                break

            # Get the most frequent pair
            pair = max(stats, key=stats.get)
            count = stats[pair]

            # Mint new token
            idx = len(self.vocab)

            # Merge in all sequences
            self.ids_list = [self.merge(ids, pair, idx) for ids in self.ids_list]

            # Update vocab and merges
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx

            print(f"merge {i + 1}/{self.VOCAB_SIZE}: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[idx]} had {count} occurrences")


        days, hrs, mins, secs = self.calculate_elapsed_time()
        logger.info(f"Training finished in {days}d {hrs}h {mins}m {secs}s")

        # ------------------------
        # Save
        self.save_models()
        logger.info("Dictionaries saved.")

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train")["text"]

    trainer = SinhalaGPETokenizerTrainer(
        dataset_size=1_000_000,
        vocab_size=5_000,
        dataset=dataset
    )
    trainer.train()
