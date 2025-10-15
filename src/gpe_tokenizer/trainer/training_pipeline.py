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

    # def merge(self, ids_idx, pair, new_id):
    #     seq = self.ids_list[ids_idx]
    #     new_seq = []
    #     changed = False
    #     i = 0
    #     while i < len(seq):
    #         if i < len(seq)-1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
    #             new_seq.append(new_id)
    #             i += 2
    #             changed = True
    #         else:
    #             new_seq.append(seq[i])
    #             i += 1
    #     if changed:
    #         self.ids_list[ids_idx] = new_seq
    #         self._updated_lists.add(ids_idx)


        

    # def multiprocess_merge(self, pair, idx):
    #     # Prepare arguments for each item in ids_list
    #     args_list = [(ids, pair, idx) for ids in self.ids_list]

    #     # Use a pool of workers
    #     with Pool(cpu_count()) as pool:
    #         # Map the merge function across all inputs
    #         self.ids_list = pool.map(self.merge_wrapper, args_list)
    


    # def get_stats(self, ids_list):
    #     self.counts = Counter() 
        
    #     for list_idx, ids in enumerate(tqdm(ids_list, desc="Counting bigrams", leave=False)):
    #         # Create bigrams using zip 
    #         bigrams = zip(ids, ids[1:]) 
    #         self.counts.update(bigrams) # count all bigrams in this list 
            
    #         # Track which list each bigram appears in 
    #         for pair in set(zip(ids, ids[1:])): # use set to avoid duplicates in same list
    #             self.lists_map[pair].add(list_idx) 
            
    #     return self.counts

    def merge(self, ids, pair, idx):
        """
        Merge a bigram pair in a sequence and mark the sequence as updated for get_stats().
        """
        seq = self.ids_list[ids]
        new_ids = []
        changed = False
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
                changed = True
            else:
                new_ids.append(seq[i])
                i += 1

        if changed:
            if not hasattr(self, "_prev_seqs"):
                self._prev_seqs = {}
            # Store the old sequence before replacing
            self._prev_seqs[ids] = seq.copy()
            self.ids_list[ids] = new_ids
            if not hasattr(self, "_updated_lists"):
                self._updated_lists = set()
            self._updated_lists.add(ids)


    def get_stats(self, ids_list):
        """
        Incrementally calculate bigram counts.
        Only recompute sequences that changed since the last merge.
        """
        if not hasattr(self, "_updated_lists"):
            # First call: all sequences are updated
            self._updated_lists = set(range(len(ids_list)))
            self.counts = Counter()
            self.lists_map = defaultdict(set)
            self._prev_seqs = {}

        # Remove old bigrams for updated lists
        for list_idx in self._updated_lists:
            old_seq = self._prev_seqs.get(list_idx, ids_list[list_idx])
            old_bigrams = list(zip(old_seq, old_seq[1:]))
            for bg in old_bigrams:
                if bg in self.counts:
                    self.counts[bg] -= 1
                    if self.counts[bg] <= 0:
                        del self.counts[bg]
                self.lists_map[bg].discard(list_idx)
                if not self.lists_map[bg]:
                    del self.lists_map[bg]

        # Recount bigrams for updated lists
        for list_idx in tqdm(self._updated_lists, desc="Counting bigrams", leave=False):
            seq = ids_list[list_idx]
            bigrams = list(zip(seq, seq[1:]))
            self.counts.update(bigrams)
            for bg in set(bigrams):
                self.lists_map[bg].add(list_idx)

        # Clear tracking for next merge
        self._updated_lists = set()
        self._prev_seqs = {}

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
        dataset_size=5_000_000,
        vocab_size=16_000,
        dataset=dataset
    )
    trainer.train()
