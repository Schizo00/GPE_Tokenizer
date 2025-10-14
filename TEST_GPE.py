import pickle
import grapheme
import numpy as np
from tqdm.auto import tqdm

# ------------------------
# Config
# ------------------------
OUTPUT_DIR = r"D:\My Studies\Research\Tokenization\LAT\train\indic_merges"

# ------------------------
# Load pickled files
# ------------------------
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

vocab = load_pickle(f"{OUTPUT_DIR}/vocab.pkl")
vocab_re = load_pickle(f"{OUTPUT_DIR}/vocab_re.pkl")
merges = load_pickle(f"{OUTPUT_DIR}/merges.pkl")

# ------------------------
# Grapheme tokenizer
# ------------------------
def tokenize_graphemes(text):
    """Split text into graphemes."""
    return list(grapheme.graphemes(text))

# ------------------------
# BPE encode function
# ------------------------
def encode_bpe(word, vocab_re, merges):
    """
    Encode a single word using the trained BPE merges.
    Returns both token IDs and token strings.
    """
    # Step 1: Convert word to grapheme IDs
    ids = [vocab_re[g] for g in tokenize_graphemes(word) if g in vocab_re]

    # Step 2: Apply merges sequentially (must follow training order)
    sorted_merges = sorted(merges.items(), key=lambda x: x[1])
    for pair, new_id in sorted_merges:
        i = 0
        new_ids = []
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids

    # Convert IDs to tokens
    tokens = [vocab[i] for i in ids]
    return ids, tokens

# ------------------------
# Decode function
# ------------------------
def decode_bpe(ids, vocab):
    return "".join(vocab[i] for i in ids)

# ------------------------
# Encode & decode full sentences
# ------------------------
def encode_sentence(sentence, vocab_re, merges):
    words = sentence.split()
    encoded = [encode_bpe(word, vocab_re, merges) for word in words]
    return encoded

def decode_sentence(encoded, vocab):
    return " ".join([decode_bpe(ids, vocab) for ids, _ in encoded])

def tokens_sentence(encoded):
    """Return the token strings for a full sentence"""
    return [tokens for _, tokens in encoded]

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
# ------------------------
# Example Sinhala words and sentences
# ------------------------
    words = [
        "ක්‍රිස්තුමාසය",
        "පරිගණකය",
        "ශ්‍රී ලංකාව",
        "විද්‍යාව",
        "සංස්කෘතිය"
    ]

    sentences = [
        "මෙය පරීක්ෂා කිරීමකි",
        "ශ්‍රී ලංකාවේ අගනගරය කොළඹ වේ",
        "කෘතිම බුද්ධිය යනු අනාගත තාක්ෂණයයි",
        "සිංහල භාෂාව ලියන රීති ඉගෙන ගන්න",
        "විද්‍යාගාරය විද්‍යාත්මක පර්යේෂණ සඳහා යොදා ගැනේ",
        "ශ්‍රී ලංකාව"
    ]

    # ------------------------
    # Test words
    # ------------------------
    print("==== Word-level BPE Encoding ====")
    for word in words:
        token_ids, tokens = encode_bpe(word, vocab_re, merges)
        reconstructed = decode_bpe(token_ids, vocab)
        print(f"Word: {word}")
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {tokens}")
        print(f"Reconstructed: {reconstructed}")
        print("-" * 50)

    # ------------------------
    # Test sentences
    # ------------------------
    print("\n==== Sentence-level BPE Encoding ====")
    for sentence in sentences:
        encoded = encode_sentence(sentence, vocab_re, merges)
        decoded = decode_sentence(encoded, vocab)
        print(f"Sentence: {sentence}")
        print("Encoded IDs & Tokens:")
        for ids, toks in encoded:
            print(ids, "->", toks)
        print(f"Decoded: {decoded}")
        print("-" * 50)