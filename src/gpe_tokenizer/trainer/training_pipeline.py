import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SinhalaGPETokenizerTrainer:
    def __init__(self, dataset_size, num_merges, filepath: str = None, dataset: list = None, output_dir: str = None):
        self.DUMMY_PREFIX = " "
        self.DATASET_SIZE = dataset_size
        self.VOCAB_SIZE = num_merges
        self.start_time = time.time()
        self.vocab = {}
        self.vocab_re = {}
        self.merges = {}
        self.output_dir = output_dir or "src/gpe_tokenizer/models"

        if filepath and dataset:
            raise ValueError("Provide either a filepath or a dataset, not both.")
        if filepath:
            from dataloader import DataLoader
            self.lines = DataLoader(filepath=filepath).load_file()
        elif dataset:
            self.lines = dataset

        self.lines = self.lines[:self.DATASET_SIZE]

    def calculate_elapsed_time(self):
        td = time.time() - self.start_time
        days, remainder = divmod(td, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Time taken: {int(days)}d {int(hours)}h {int(minutes)}m")
        return int(days), int(hours), int(minutes), int(seconds)

    def dict_to_pickle(self, dictionary, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(dictionary, f)
        logger.info(f"Saved: {file_path}")

    def save_trained_model(self):
        self.dict_to_pickle(self.merges, os.path.join(self.output_dir, "merges.pkl"))
        self.dict_to_pickle(self.vocab, os.path.join(self.output_dir, "vocab.pkl"))
        self.dict_to_pickle(self.vocab_re, os.path.join(self.output_dir, "vocab_re.pkl"))

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def get_stats(self, ids_list):
        counts = {}
        for ids in ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @lru_cache(maxsize=2**18)
    def tokenize_graphemes_cached(self, word):
        return list(grapheme.graphemes(word))

    def process_dataset(self):
        for line in tqdm(self.lines, total=self.DATASET_SIZE, desc="Loading dataset"):
            yield self.DUMMY_PREFIX + re.sub(r'\s+', ' ', line.strip())

    def build_grapheme_vocab(self):
        logger.info("Building initial grapheme vocabulary...")
        idx = 0
        for line in self.process_dataset():
            for word in line.split():
                for g in self.tokenize_graphemes_cached(word):
                    if g not in self.vocab_re:
                        self.vocab[idx] = g
                        self.vocab_re[g] = idx
                        idx += 1
        logger.info(f"Initial grapheme vocab size: {len(self.vocab)}")

    def convert_text_to_ids(self):
        logger.info("Converting text to ID sequences...")
        ids_list = []
        for line in self.process_dataset():
            for word in line.split():
                ids = [self.vocab_re[g] for g in self.tokenize_graphemes_cached(word)]
                ids_list.append(ids)
        return ids_list

    def train(self):
        # Build vocab
        self.build_grapheme_vocab()
        # Convert to ID sequences
        self.ids_list = self.convert_text_to_ids()
        # Free memory
        del self.lines

        # BPE training
        pbar = tqdm(total=self.VOCAB_SIZE, desc="BPE merges")
        for i in range(self.VOCAB_SIZE):
            stats = self.get_stats(self.ids_list)
            if not stats:
                logger.info("No more pairs to merge.")
                break
            pair = max(stats, key=stats.get)
            count = stats[pair]
            idx = len(self.vocab)
            self.ids_list = [self.merge(ids, pair, idx) for ids in self.ids_list]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx
            pbar.update()
        pbar.close()

        # Timing
        self.calculate_elapsed_time()
        # Save models
        self.save_trained_model()
        logger.info("Training complete.")

if __name__ == "__main__":
    from datasets import load_dataset
    trainer = SinhalaGPETokenizerTrainer(
        dataset_size=5000000,
        num_merges=32000,
        dataset=load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train")["text"]
    )
    trainer.train()
