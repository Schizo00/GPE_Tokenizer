import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
from dataloader import DataLoader
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class SinhalaGPETokenizerTrainer:

    def __init__(self, dataset_size, num_merges, filepath: str = None, dataset: str = None):
        self.DUMMY_PREFIX: str = " "
        self.DATASET_SIZE: int = dataset_size
        self.VOCAB_SIZE: int = num_merges
        self.start_time: float = time.time()
        self.vocab: dict = {}
        self.vocab_re: dict = {}
        self.merges: dict = {}
        self.output_dir: str = r"src/gpe_tokenizer/models"
        self.lines = None

        if filepath and dataset:
            raise ValueError("Provide either a filepath or a dataset, not both.")
        if filepath:
            self.lines = DataLoader(filepath=filepath).load_file()
        elif dataset:
            self.lines = dataset

        self.lines = self.lines[:self.DATASET_SIZE]

        # Cache grapheme tokenization
        self._grapheme_cache = {}

    def calculate_elapsed_time(self):
        td = time.time() - self.start_time
        days, remainder = divmod(td, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f'Time taken: {int(days)}d {int(hours)}h {int(minutes)}m')
        return int(days), int(hours), int(minutes), int(seconds)

    def dict_to_pickle(self, dictionary, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f)
        logger.info(f"Dictionary saved to {file_path}")

    def save_trained_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.dict_to_pickle(self.merges, os.path.join(self.output_dir, "merges.pkl"))
        self.dict_to_pickle(self.vocab, os.path.join(self.output_dir, "vocab.pkl"))
        self.dict_to_pickle(self.vocab_re, os.path.join(self.output_dir, "vocab_re.pkl"))

    @staticmethod
    @lru_cache(maxsize=500_000)
    def merge_cached(ids_tuple, pair, idx):
        """Cached merge function using tuples as input."""
        new_ids = []
        i = 0
        while i < len(ids_tuple):
            if i < len(ids_tuple) - 1 and ids_tuple[i] == pair[0] and ids_tuple[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids_tuple[i])
                i += 1
        return tuple(new_ids)

    def merge(self, ids, pair, idx):
        return list(self.merge_cached(tuple(ids), pair, idx))

    def get_stats(self, ids_list):
        counts = {}
        for ids in ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def tokenize_graphemes(self, word):
        if word not in self._grapheme_cache:
            self._grapheme_cache[word] = list(grapheme.graphemes(word))
        return self._grapheme_cache[word]

    def build_grapheme_vocab(self):
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
        logger.info(f"Initial grapheme vocab size: {len(self.vocab)}")

    def convert_text_to_ids(self, batch_size=10000):
        self.ids_list = []
        for batch_start in tqdm(range(0, len(self.lines), batch_size), desc="Converting text to IDs"):
            batch = self.lines[batch_start: batch_start + batch_size]
            for line in batch:
                words = line.split()
                for word in words:
                    ids = [self.vocab_re.setdefault(g, len(self.vocab_re)) for g in self.tokenize_graphemes(word)]
                    self.ids_list.append(ids)

    def train(self):
        self.build_grapheme_vocab()
        self.convert_text_to_ids()
        
        del self.lines
        # BPE Training
        for i in tqdm(range(self.VOCAB_SIZE), desc="BPE merges"):
            stats = self.get_stats(self.ids_list)
            if not stats:
                logger.info("No more pairs to merge.")
                break

            pair = max(stats, key=stats.get)
            idx = len(self.vocab)
            self.ids_list = [self.merge(ids, pair, idx) for ids in self.ids_list]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx

            tqdm.write(f"Merge {i+1}/{self.VOCAB_SIZE}: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[idx]} | count {stats[pair]}")

        self.calculate_elapsed_time()
        self.save_trained_model()
        logger.info("Training complete.")


if __name__ == "__main__":
    from datasets import load_dataset
    dataset = load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train")["text"]

    trainer = SinhalaGPETokenizerTrainer(
        dataset_size=5_000_000,
        num_merges=32_000,
        dataset=dataset
    )
    trainer.train()
