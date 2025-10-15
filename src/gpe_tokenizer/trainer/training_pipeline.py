import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
import numpy as np
from collections import Counter, defaultdict
import logging
import itertools
from multiprocessing import Pool, cpu_count




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
        self.graphemes_list = []
        self.lists_map = defaultdict(set)
        self.output_dir = output_dir
        self.counts = Counter()

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

    def merge_wrapper(self, args):
        ids, pair, idx = args
        # Assuming `self.merge` is a method; you can pass the object if needed
        return self.merge(ids, pair, idx) 

    def merge(self, ids, pair, idx):
        """
        ids_lists: list of lists of ids
        pair: tuple to merge, e.g., (1,2)
        idx: new id to replace the pair
        lists_map: dict mapping bigram -> set of list indices where it occurs
        """
        if pair not in self.lists_map:
            return  # nothing to merge

        new_ids = []
        i = 0
        while i < len(self.ids_list[ids]):
            if i < len(self.ids_list[ids]) - 1 and self.ids_list[ids][i] == pair[0] and self.ids_list[ids][i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2  # skip the pair
            else:
                new_ids.append(self.ids_list[ids][i])
                i += 1
                
            self.ids_list[ids] = new_ids  # update only this list


        

    # def multiprocess_merge(self, pair, idx):
    #     # Prepare arguments for each item in ids_list
    #     args_list = [(ids, pair, idx) for ids in self.ids_list]

    #     # Use a pool of workers
    #     with Pool(cpu_count()) as pool:
    #         # Map the merge function across all inputs
    #         self.ids_list = pool.map(self.merge_wrapper, args_list)
    

    def get_stats(self, ids_list):

        self.counts = Counter()

        for list_idx, ids in enumerate(tqdm(ids_list, desc="Counting bigrams", leave=False)):
            # Create bigrams using zip
            bigrams = zip(ids, ids[1:])
            self.counts.update(bigrams)  # count all bigrams in this list
            # Track which list each bigram appears in
            for pair in set(zip(ids, ids[1:])):  # use set to avoid duplicates in same list
                self.lists_map[pair].add(list_idx)

        return self.counts


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
                self.ids_list.append([self.vocab_re[g] for g in self.tokenize_graphemes(word)])



    # ------------------------
    # Train BPE
    # ------------------------
    def train(self):
        logger.info("Starting training...")
        self.build_vocab()
        self.convert_to_ids()

        del self.lines # Free up memory

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
            # [self.merge(ids, pair, idx) for ids in tqdm(self.lists_map[pair], desc="Merging pairs", leave=False)]

            merge_bar = tqdm(total=len(self.lists_map[pair]), desc="Merging pair", leave=False)

            for ids in list(self.lists_map[pair]):
                self.merge(ids, pair, idx)
                merge_bar.update(1)

            merge_bar.close()
            # self.multiprocess_merge(pair, idx)
            
            self.lists_map[pair].pop() # Pop the pair from lists map since its not needed anymore

            # Update vocab and merges
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx

            tqdm.write(f"merge {i + 1}/{self.VOCAB_SIZE}: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[idx]} had {count} occurrences")


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
        dataset_size=10_000_00,
        vocab_size=5_000,
        dataset=dataset
    )
    trainer.train()
