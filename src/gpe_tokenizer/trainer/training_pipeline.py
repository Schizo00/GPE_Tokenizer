import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SinhalaGPETokenizerTrainer:

    def __init__(self, dataset_size, num_merges, filepath=None, dataset=None, output_dir="models"):
        self.DUMMY_PREFIX = " "
        self.DATASET_SIZE = dataset_size
        self.NUM_MERGES = num_merges
        self.start_time = time.time()
        self.vocab = {}
        self.vocab_re = {}
        self.merges = {}
        self.grapheme_cache = {}
        self.output_dir = output_dir

        if filepath and dataset:
            raise ValueError("Provide either filepath or dataset, not both.")
        
        if filepath:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.lines = f.readlines()
        elif dataset:
            self.lines = dataset

        self.lines = self.lines[:self.DATASET_SIZE]

    # ------------------------
    # Utilities
    # ------------------------
    def elapsed_time(self):
        td = time.time() - self.start_time
        days, rem = divmod(td, 86400)
        hrs, rem = divmod(rem, 3600)
        mins, secs = divmod(rem, 60)
        return int(days), int(hrs), int(mins), int(secs)

    def save_pickle(self, obj, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    # ------------------------
    # Grapheme tokenization with caching
    # ------------------------
    def tokenize_graphemes(self, word):
        if word in self.grapheme_cache:
            return self.grapheme_cache[word]
        tokens = list(grapheme.graphemes(word))
        self.grapheme_cache[word] = tokens
        return tokens

    # ------------------------
    # Build initial vocab
    # ------------------------
    def build_vocab(self):
        self.graphemes_list = []
        for line in tqdm(self.lines, desc="Building vocab"):
            words = line.split()
            for word in words:
                for g in self.tokenize_graphemes(word):
                    if g not in self.vocab_re:
                        idx = len(self.graphemes_list)
                        self.graphemes_list.append(g)
                        self.vocab[idx] = g
                        self.vocab_re[g] = idx
        logger.info(f"Initial vocab size: {len(self.vocab)}")

    # ------------------------
    # Convert text to ID sequences
    # ------------------------
    def convert_to_ids(self):
        self.ids_list = []
        for line in tqdm(self.lines, desc="Converting text to IDs"):
            words = line.split()
            for word in words:
                ids = [self.vocab_re[g] for g in self.tokenize_graphemes(word)]
                self.ids_list.append(np.array(ids, dtype=np.int32))

    # ------------------------
    # Incremental BPE
    # ------------------------
    def get_pair_counts(self):
        pair_counts = defaultdict(int)
        pair_positions = defaultdict(list)
        for seq_idx, seq in enumerate(self.ids_list):
            if len(seq) < 2:
                continue
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_counts[pair] += 1
                pair_positions[pair].append((seq_idx, i))
        return pair_counts, pair_positions

    def merge_pair(self, pair, idx, pair_positions):
        for seq_idx, pos in pair_positions[pair]:
            seq = self.ids_list[seq_idx]
            if pos >= len(seq) - 1:
                continue
            if seq[pos] == pair[0] and seq[pos + 1] == pair[1]:
                seq[pos] = idx
                self.ids_list[seq_idx] = np.delete(seq, pos + 1)
        del pair_positions[pair]

    # ------------------------
    # Train BPE
    # ------------------------
    def train(self):
        self.build_vocab()
        self.convert_to_ids()

        pbar = tqdm(total=self.NUM_MERGES, desc="BPE merges")
        for i in range(self.NUM_MERGES):
            pair_counts, pair_positions = self.get_pair_counts()
            if not pair_counts:
                logger.info("No more pairs to merge!")
                break

            # Most frequent pair
            pair = max(pair_counts, key=pair_counts.get)
            count = pair_counts[pair]

            # Skip rare pairs
            if count < 2:
                logger.info("Skipping rare pair with count < 2")
                continue

            # Mint new token
            idx = len(self.vocab)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx

            self.merge_pair(pair, idx, pair_positions)

            tqdm.write(f"merge {i+1}/{self.NUM_MERGES}: {self.vocab[pair[0]]}+{self.vocab[pair[1]]} -> {self.vocab[idx]} | freq={count}")
            pbar.update()

        days, hrs, mins = self.elapsed_time()
        logger.info(f"Training finished in {days}d {hrs}h {mins}m")

        # ------------------------
        # Save
        self.save_pickle(self.vocab, os.path.join(self.output_dir, "vocab.pkl"))
        self.save_pickle(self.vocab_re, os.path.join(self.output_dir, "vocab_re.pkl"))
        self.save_pickle(self.merges, os.path.join(self.output_dir, "merges.pkl"))
        logger.info("Dictionaries saved.")

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train")["text"]

    trainer = SinhalaGPETokenizerTrainer(
        dataset_size=5000000,
        num_merges=32000,
        dataset=dataset,
        output_dir="./models"
    )
    trainer.train()
