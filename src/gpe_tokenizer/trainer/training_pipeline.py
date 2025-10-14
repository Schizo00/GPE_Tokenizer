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

    def __init__(self, dataset_size, num_merges, filepath: str = None, dataset: list = None):
        self.DUMMY_PREFIX = " "
        self.DATASET_SIZE = dataset_size
        self.VOCAB_SIZE = num_merges
        self.start_time = time.time()
        self.vocab = {}
        self.vocab_re = {}
        self.merges = {}
        self.output_dir = r"src/gpe_tokenizer/models"

        if filepath and dataset:
            raise ValueError("Provide either filepath or dataset, not both.")
        if filepath:
            from dataloader import DataLoader
            self.lines = DataLoader(filepath=filepath).load_file()
        else:
            self.lines = dataset

        self.lines = self.lines[:self.DATASET_SIZE]

    def calculate_elapsed_time(self):
        td = time.time() - self.start_time
        days, rem = divmod(td, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(f'Time taken: {int(days)}d {int(hours)}h {int(minutes)}m')
        return int(days), int(hours), int(minutes), int(seconds)

    def dict_to_pickle(self, dictionary, file_path):
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(dictionary, f)
            logger.info(f"Saved {file_path}")
        except Exception as e:
            logger.error(f"Error saving {file_path}: {e}")

    def save_trained_model(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self.dict_to_pickle(self.merges, os.path.join(self.output_dir, "merges.pkl"))
        self.dict_to_pickle(self.vocab, os.path.join(self.output_dir, "vocab.pkl"))
        self.dict_to_pickle(self.vocab_re, os.path.join(self.output_dir, "vocab_re.pkl"))

    @lru_cache(maxsize=1_000_000)
    def merge_cached(self, ids_tuple, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids_tuple):
            if i < len(ids_tuple) - 1 and ids_tuple[i] == pair[0] and ids_tuple[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids_tuple[i])
                i += 1
        return tuple(new_ids)

    @lru_cache(maxsize=1_000_000)
    def grapheme_cached(self, word):
        return tuple(grapheme.graphemes(word))

    def get_stats(self, ids_list):
        counts = {}
        for ids in ids_list:
            for a, b in zip(ids, ids[1:]):
                counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    def process_dataset(self):
        self.lines = [self.DUMMY_PREFIX + re.sub(r'\s+', ' ', line.strip()) for line in self.lines]

    def build_grapheme_vocab(self):
        self.graphemes_list = []
        for line in tqdm(self.lines, desc="[Tokenizer] Building vocab"):
            for word in line.split():
                for g in self.grapheme_cached(word):
                    if g not in self.vocab_re:
                        idx = len(self.graphemes_list)
                        self.graphemes_list.append(g)
                        self.vocab[idx] = g
                        self.vocab_re[g] = idx
        logger.info(f"Initial grapheme vocab size: {len(self.vocab)}")

    def convert_text_to_ids(self, batch_size=10000):
        self.ids_list = []
        for i in tqdm(range(0, len(self.lines), batch_size), desc="[Tokenizer] Converting text to IDs"):
            batch = self.lines[i:i+batch_size]
            for line in batch:
                for word in line.split():
                    self.ids_list.append([self.vocab_re[g] for g in self.grapheme_cached(word)])

    def train(self):
        self.process_dataset()
        self.build_grapheme_vocab()
        self.convert_text_to_ids()

        pbar = tqdm(total=self.VOCAB_SIZE, desc="[Tokenizer] Training BPE merges")

        for i in range(self.VOCAB_SIZE):
            stats = self.get_stats(self.ids_list)
            if not stats:
                logger.info("No more pairs to merge.")
                break

            # Most frequent pair
            pair = max(stats, key=stats.get)
            count = stats[pair]

            # Skip if pair never occurs
            if count == 0:
                tqdm.write(f"Skipping merge {i+1}: pair {pair} never occurs")
                continue

            idx = len(self.vocab)

            # Merge sequences
            new_ids_list = []
            for ids in self.ids_list:
                if pair[0] in ids and pair[1] in ids:
                    new_ids_list.append(list(self.merge_cached(tuple(ids), pair, idx)))
                else:
                    new_ids_list.append(ids)
            self.ids_list = new_ids_list

            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.vocab_re[self.vocab[idx]] = idx

            # Estimate remaining meaningful merges
            total_pairs = sum(stats.values())
            remaining_ratio = count / total_pairs if total_pairs else 0
            tqdm.write(f"merge {i+1}/{self.VOCAB_SIZE}: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[idx]} | freq: {count} | remaining_ratio: {remaining_ratio:.4f}")

            pbar.update()

        self.calculate_elapsed_time()
        self.save_trained_model()
        logger.info("Training complete.")


if __name__ == "__main__":
    from datasets import load_dataset

    trainer = SinhalaGPETokenizerTrainer(
        dataset_size=5_000_000,
        num_merges=32_000,
        dataset=load_dataset("polyglots/MADLAD_CulturaX_cleaned", split="train")["text"]
    )
    trainer.train()
