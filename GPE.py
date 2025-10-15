import regex as re
import grapheme
import os
import pickle
import time
from tqdm.auto import tqdm
from src.sinhala_tokenizer_analysis.dataloaders import ContinualPretrainingDataLoader

# ------------------------
# Configuration
# ------------------------
loader = ContinualPretrainingDataLoader()
DUMMY_PREFIX = " "  # leading space for word boundary
DATASET_SIZE = 10_000_0
NUM_MERGES = 1000

# ------------------------
# Timing helper
# ------------------------
def calculate_elapsed_time(start_time):
    end_time = time.time()
    td = end_time - start_time
    days, remainder = divmod(td, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(days), int(hours), int(minutes), int(seconds)

# ------------------------
# Save dictionary helper
# ------------------------
def save_dict_to_pickle(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

# ------------------------
# Merge helper
# ------------------------
def merge(ids, pair, idx):
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

# ------------------------
# Count consecutive pairs
# ------------------------
def get_stats(ids_list):
    counts = {}
    for ids in ids_list:
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts

# ------------------------
# Grapheme tokenizer
# ------------------------
def tokenize_graphemes(text):
    """Return list of graphemes for a word."""
    return list(grapheme.graphemes(text))

# ------------------------
# Load dataset
# ------------------------
lines = [line for line in loader.load_dataset().select(range(DATASET_SIZE))['text']]
lines = [DUMMY_PREFIX + re.sub(r'\s+', ' ', line.strip()) for line in lines]

# ------------------------
# Build initial vocab (graphemes only)
# ------------------------
vocab = {}
vocab_re = {}
graphemes_list = []

for line in tqdm(lines, desc="Building vocab"):
    words = line.split()
    for word in words:
        for g in tokenize_graphemes(word):
            if g not in vocab_re:
                idx = len(graphemes_list)
                graphemes_list.append(g)
                vocab[idx] = g
                vocab_re[g] = idx

# ------------------------
# Convert lines to grapheme ID sequences (per word)
# ------------------------
ids_list = []
for line in tqdm(lines, desc="Converting text to IDs"):
    words = line.split()
    for word in words:
        ids_list.append([vocab_re[g] for g in tokenize_graphemes(word)])

# ------------------------
# BPE training
# ------------------------
merges = {}
start_time = time.time()

for i in range(NUM_MERGES):
    stats = get_stats(ids_list)
    if not stats:
        print("No more pairs to merge!")
        break

    # Get the most frequent pair
    pair = max(stats, key=stats.get)
    count = stats[pair]

    # Mint new token
    idx = len(vocab)

    # Merge in all sequences
    ids_list = [merge(ids, pair, idx) for ids in ids_list]

    # Update vocab and merges
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
    vocab_re[vocab[idx]] = idx

    print(f"merge {i + 1}/{NUM_MERGES}: {vocab[pair[0]]} + {vocab[pair[1]]} -> {vocab[idx]} had {count} occurrences")

# ------------------------
# Timing
# ------------------------
days, hours, minutes, _ = calculate_elapsed_time(start_time)
print(f'Time taken: {days} days {hours} hrs {minutes} mins')

# ------------------------
# Save results
# ------------------------
output_dir = r"D:\My Studies\Research\Tokenization\LAT\train\indic_merges"
os.makedirs(output_dir, exist_ok=True)

save_dict_to_pickle(merges, os.path.join(output_dir, "merges.pkl"))
save_dict_to_pickle(vocab, os.path.join(output_dir, "vocab.pkl"))
save_dict_to_pickle(vocab_re, os.path.join(output_dir, "vocab_re.pkl"))

print("Training finished and dictionaries saved.")
