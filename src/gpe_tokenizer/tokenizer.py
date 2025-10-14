import pickle
import grapheme
from pathlib import Path

class SinhalaGPETokenizer:
    def __init__(self, models_dir="models"):
        models_dir = Path(models_dir)
        self.vocab = pickle.load(open(models_dir / "vocab.pkl", "rb"))
        self.vocab_re = pickle.load(open(models_dir / "vocab_re.pkl", "rb"))
        self.merges = pickle.load(open(models_dir / "merges.pkl", "rb"))
        self.max_id = max(self.vocab.keys())

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

    def encode_word(self, word):
        ids = []
        for g in grapheme.graphemes(word):
            if g not in self.vocab_re:
                # Add grapheme dynamically
                self.max_id += 1
                self.vocab[self.max_id] = g
                self.vocab_re[g] = self.max_id
            ids.append(self.vocab_re[g])
        # Apply BPE merges
        for pair, idx in self.merges.items():
            ids = self.merge(ids, pair, idx)
        return ids

    def encode(self, text):
        encoded = []
        for word in text.split():
            encoded.extend(self.encode_word(word))
            if " " in self.vocab_re:
                encoded.append(self.vocab_re[" "])
        if encoded and " " in self.vocab_re:
            encoded = encoded[:-1]
        return encoded

    def decode(self, ids):
        return "".join([self.vocab[i] for i in ids])

    def tokens(self, text):
        return [self.vocab[i] for i in self.encode(text)]
    

if __name__ == "__main__":
    from tokenizer import SinhalaGPETokenizer
    tokenizer = SinhalaGPETokenizer(models_dir="./src/gpe_tokenizer/models")

    def visualize_graphemes(text):
        print("Text:", text)
        encoded = tokenizer.encode(text)
        print("Encoded:", encoded)
        decoded = tokenizer.decode(encoded)
        print("Decoded:", decoded)
        tokens = tokenizer.tokens(text)
        print("Tokens:", tokens)


    sentences = [
        "මෙය පරීක්ෂා කිරීමකි",
        "ශ්‍රී ලංකාවේ අගනගරය කොළඹ වේ",
        "කෘතිම බුද්ධිය යනු අනාගත තාක්ෂණයයි",
        "සිංහල භාෂාව ලියන රීති ඉගෙන ගන්න",
        "විද්‍යාගාරය විද්‍යාත්මක පර්යේෂණ සඳහා යොදා ගැනේ",
        "ශ්‍රී ලංකාව"
    ]

    for sentence in sentences:
        visualize_graphemes(sentence)
        print()
