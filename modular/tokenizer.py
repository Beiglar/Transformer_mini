from collections import Counter
import json

class MiniCharTok:
    """
    A simple character-level tokenizer with frequency-based vocabulary and special tokens.

    This tokenizer builds its vocabulary from the provided text, filtering characters
    by minimum frequency. Rare characters are replaced with <UNK>. Optionally adds <PAD>.

    Args:
        special (dict[str, int]): Mapping of special token strings to their IDs.
                                  Must include at least '<UNK>' and optionally '<PAD>'.
        min_freq (int): Minimum frequency for a character to be included in vocab.
                        Characters below this threshold are replaced with <UNK>.
    """

    def __init__(self, special: dict[str, int], min_freq: int = 1) -> None:
        self.min_freq = min_freq
        self.special_tokens = special.copy()  # e.g., {'<PAD>': 0, '<UNK>': 1}
        self.pad_token_id = special.get('<PAD>', None)
        self.unk_token_id = special.get('<UNK>', None)

        if self.unk_token_id is None:
            raise ValueError("special must contain '<UNK>' token ID")
        
        self.chars = []           # List of frequent characters (excluding special tokens)
        self.char_to_int = {}     # char -> id (only for frequent chars)
        self.int_to_char = {}     # id -> char (includes all special + frequent chars)
        self.vocab_size = 0

        # Initialize special tokens in vocab
        for token, tid in self.special_tokens.items():
            self.int_to_char[tid] = token

        # Special tokens must have unique, low IDs (typically 0, 1, ...)
        self._ensure_special_ids_low()

    def _ensure_special_ids_low(self):
        """Ensure special token IDs are the lowest in the vocab."""
        special_ids = set(self.special_tokens.values())
        if min(special_ids) != 0:
            raise ValueError("Special token IDs must start from 0 or low values for consistent encoding.")

    def __call__(self, text: str) -> None:
        """
        Builds the tokenizer's vocabulary from the given text.
        Characters appearing less than min_freq are replaced with <UNK>.
        """
        # Count character frequencies
        char_counts = Counter(text)

        # Build vocabulary: only include chars with freq >= min_freq
        self.chars = sorted([ch for ch, cnt in char_counts.items() if cnt >= self.min_freq])

        # Map frequent chars to IDs (starting after special tokens)
        start_idx = len(self.special_tokens)
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars, start=start_idx)}
        self.int_to_char.update({i: ch for ch, i in self.char_to_int.items()})

        # Update vocab size
        self.vocab_size = len(self.int_to_char)

        # Validate: <UNK> must be in vocab
        if self.unk_token_id not in self.int_to_char:
            self.int_to_char[self.unk_token_id] = '<UNK>'  # Ensure it's there

    def Encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.
        Characters not in vocab (below min_freq) are mapped to <UNK>.
        """
        result = []
        for char in text:
            if char in self.char_to_int:
                result.append(self.char_to_int[char])
            else:
                result.append(self.unk_token_id)
        return result

    def Decode(self, char_ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        Unknown IDs are replaced with empty string (or you could use <UNK> symbol).
        """
        return "".join(self.int_to_char.get(char_id, '') for char_id in char_ids)

    def GetPieceSize(self) -> int:
        """Returns the total size of the vocabulary (including special tokens)."""
        return self.vocab_size

    def get_special_tokens(self) -> dict[str, int]:
        """Returns a copy of the special tokens mapping."""
        return self.special_tokens.copy()

    def __repr__(self) -> str:
        return (f"MiniCharTok(vocab_size={self.vocab_size}, "
                f"min_freq={self.min_freq}, "
                f"special_tokens={self.special_tokens})")

    def save(self, filepath: str) -> None:
        data = {
            'special_tokens': self.special_tokens,
            'min_freq': self.min_freq,
            'chars': self.chars,
            'vocab_size': self.vocab_size,
            'char_to_int': self.char_to_int,
            'int_to_char': self.int_to_char
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        tokenizer = cls(data['special_tokens'], data['min_freq'])
        tokenizer.chars = data['chars']
        tokenizer.vocab_size = data['vocab_size']
        tokenizer.char_to_int = data['char_to_int']
        tokenizer.int_to_char = {int(k): v for k, v in data['int_to_char'].items()}
        return tokenizer
