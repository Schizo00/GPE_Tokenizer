import pickle
import grapheme
from pathlib import Path
from typing import List, Optional, Dict, Any
from transformers import PreTrainedTokenizer


class SinhalaGPETokenizer(PreTrainedTokenizer):
    """
    Sinhala Grapheme-based BPE Tokenizer compatible with Hugging Face Transformers.
    Supports BERT, LLaMA, GPT, and other transformer architectures.
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        models_dir="gpe_tokenizer/models",
        model_type="llama",  # "llama", "bert", or "gpt"
        unk_token=None,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        mask_token=None,
        cls_token=None,
        sep_token=None,
        **kwargs
    ):
        """
        Initialize the tokenizer with model-specific special tokens.
        
        Args:
            models_dir: Directory containing vocab.pkl, vocab_re.pkl, and merges.pkl
            model_type: Type of model ("llama", "bert", "gpt")
            Special tokens can be overridden; defaults are set based on model_type
        """
        self.models_dir = Path(models_dir)
        self.model_type = model_type.lower()
        print(self.models_dir)
        # Load vocabulary and merges
        self.vocab = pickle.load(open(self.models_dir / "vocab.pkl", "rb"))
        self.vocab_re = pickle.load(open(self.models_dir / "vocab_re.pkl", "rb"))
        self.merges = pickle.load(open(self.models_dir / "merges.pkl", "rb"))
        self.max_id = max(self.vocab.keys())
        
        # Set default special tokens based on model type
        if self.model_type == "bert":
            unk_token = unk_token or "[UNK]"
            pad_token = pad_token or "[PAD]"
            cls_token = cls_token or "[CLS]"
            sep_token = sep_token or "[SEP]"
            mask_token = mask_token or "[MASK]"
            bos_token = bos_token or cls_token
            eos_token = eos_token or sep_token
        elif self.model_type == "llama":
            unk_token = unk_token or "<unk>"
            bos_token = bos_token or "<s>"
            eos_token = eos_token or "</s>"
            pad_token = pad_token or "<pad>"  # LLaMA doesn't have pad by default
        else:  # gpt or other
            unk_token = unk_token or "<|endoftext|>"
            bos_token = bos_token or "<|endoftext|>"
            eos_token = eos_token or "<|endoftext|>"
            pad_token = pad_token or "<|endoftext|>"
        
        # Add special tokens to vocabulary
        special_tokens_map = {
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "pad_token": pad_token,
        }
        
        if mask_token:
            special_tokens_map["mask_token"] = mask_token
        if cls_token:
            special_tokens_map["cls_token"] = cls_token
        if sep_token:
            special_tokens_map["sep_token"] = sep_token
        
        for token_name, token_value in special_tokens_map.items():
            if token_value and token_value not in self.vocab_re:
                self.max_id += 1
                self.vocab[self.max_id] = token_value
                self.vocab_re[token_value] = self.max_id
        
        # Create reverse vocabulary (id -> token)
        self.ids_to_tokens = {v: k for k, v in self.vocab_re.items()}
        
        # Call parent class constructor
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            **kwargs
        )
    
    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)
    
    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary of token to index."""
        return self.vocab_re.copy()
    
    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Tokenize a string into tokens."""
        tokens = []
        for word in text.split():
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            if " " in self.vocab_re:
                tokens.append(" ")
        
        # Remove trailing space token
        if tokens and tokens[-1] == " ":
            tokens = tokens[:-1]
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using grapheme-based BPE."""
        # Convert word to token IDs
        ids = []
        for g in grapheme.graphemes(word):
            if g not in self.vocab_re:
                # Add grapheme dynamically
                self.max_id += 1
                self.vocab[self.max_id] = g
                self.vocab_re[g] = self.max_id
                self.ids_to_tokens[self.max_id] = g
            ids.append(self.vocab_re[g])
        
        # Apply BPE merges
        for pair, idx in self.merges.items():
            ids = self._merge(ids, pair, idx)
        
        # Convert IDs back to tokens
        return [self.vocab[i] for i in ids]
    
    def _merge(self, ids: List[int], pair: tuple, idx: int) -> List[int]:
        """Merge a pair of tokens."""
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
    
    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) to an id using the vocabulary."""
        return self.vocab_re.get(token, self.vocab_re.get(self.unk_token))
    
    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) to a token (str) using the vocabulary."""
        return self.vocab.get(index, self.unk_token)
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens into a single string."""
        return " ".join(tokens)
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """
        Save the tokenizer vocabulary to a directory.
        
        Returns:
            A tuple of file paths where the vocabulary was saved.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        prefix = filename_prefix + "-" if filename_prefix else ""
        
        vocab_file = save_directory / f"{prefix}vocab.pkl"
        vocab_re_file = save_directory / f"{prefix}vocab_re.pkl"
        merges_file = save_directory / f"{prefix}merges.pkl"
        
        with open(vocab_file, "wb") as f:
            pickle.dump(self.vocab, f)
        
        with open(vocab_re_file, "wb") as f:
            pickle.dump(self.vocab_re, f)
        
        with open(merges_file, "wb") as f:
            pickle.dump(self.merges, f)
        
        return (str(vocab_file), str(vocab_re_file), str(merges_file))
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs by adding special tokens based on model type.
        
        BERT format:
        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`
        
        LLaMA/GPT format:
        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s> B </s>`
        """
        if self.model_type == "bert":
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            
            if token_ids_1 is None:
                return cls + token_ids_0 + sep
            return cls + token_ids_0 + sep + token_ids_1 + sep
        else:
            bos = [self.bos_token_id]
            eos = [self.eos_token_id]
            
            if token_ids_1 is None:
                return bos + token_ids_0 + eos
            return bos + token_ids_0 + eos + token_ids_1 + eos
    
    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Returns a mask where special tokens are marked with 1 and regular tokens with 0.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        
        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]
    
    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Create token type IDs for sequence pairs (used in BERT).
        
        Returns:
            List of zeros for single sequence, or zeros for first sequence
            and ones for second sequence.
        """
        if self.model_type == "bert":
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            
            if token_ids_1 is None:
                return len(cls + token_ids_0 + sep) * [0]
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
        else:
            bos = [self.bos_token_id]
            eos = [self.eos_token_id]
            
            if token_ids_1 is None:
                return len(bos + token_ids_0 + eos) * [0]
            return len(bos + token_ids_0 + eos) * [0] + len(token_ids_1 + eos) * [1]
        

    def visualize_tokens(self, text: str) -> None:
        """
        Visualize the tokenization process with detailed output.
        """
        print(f"Original Text: {text}")
        print("-" * 80)
        
        # Get tokens
        tokens = self.tokenize(text)
        print(f"Tokens ({len(tokens)}): {tokens}")
        print("-" * 80)
        
        # Get token IDs
        token_ids = self.encode(text, add_special_tokens=False)
        print(f"Token IDs (no special): {token_ids}")
        
        # Get token IDs with special tokens
        token_ids_special = self.encode(text, add_special_tokens=True)
        print(f"Token IDs (with special): {token_ids_special}")
        print("-" * 80)
        
        # Show token-to-ID mapping
        print("Token -> ID mapping:")
        for token, token_id in zip(tokens, token_ids):
            display_token = token.replace("▁", "␣")  # Use visible space symbol
            print(f"  '{display_token}' -> {token_id}")
        print("-" * 80)
        
        # Decode
        decoded = self.decode(token_ids_special, skip_special_tokens=False)
        decoded_no_special = self.decode(token_ids_special, skip_special_tokens=True)
        
        print(f"Decoded (with special): {decoded}")
        print(f"Decoded (no special): {decoded_no_special}")
        print(f"Match original: {decoded_no_special == text}")
        print("=" * 80)




if __name__ == "__main__":
    # from tokenizer import SinhalaGPETokenizer

    # Initialize tokenizer for BERT
    tokenizer = SinhalaGPETokenizer(models_dir="src/gpe_tokenizer/models", model_type="bert")
        
    # Test tokenization

    # def test_tokens(text):
    #     encoded = tokenizer(text, return_tensors="pt")
    #     print(f"\nText: {text}")
    #     print(f"Input IDs: {encoded['input_ids']}")
    #     print(f"Decoded: {tokenizer.decode(encoded['input_ids'][0])}")

    sentences = [
        "මෙය පරීක්ෂා කිරීමකි",
        "ශ්‍රී ලංකාවේ අගනගරය කොළඹ වේ",
        "කෘතිම බුද්ධිය යනු අනාගත තාක්ෂණයයි",
        "සිංහල භාෂාව ලියන රීති ඉගෙන ගන්න",
        "විද්‍යාගාරය විද්‍යාත්මක පර්යේෂණ සඳහා යොදා ගැනේ",
        "ශ්‍රී ලංකාව"
    ]

    for sent in sentences:
        tokenizer.visualize_tokens(sent)
