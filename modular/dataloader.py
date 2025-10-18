import os
import queue
import threading
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Generator, Iterator
from modular.tokenizer import MiniCharTok

Array = jax.Array

class MemmapDataLoader:
    """
    A class to generate batches of input and target sequences from a memory-mapped
    NumPy array of token IDs for sequence model training.

    Args:
        token_ids_path (str): Path to the .npy or .bin file containing the 1D array of token IDs.
        max_seq_len (int): Maximum sequence length for inputs and targets.
        batch_size (int): Number of sequences per batch.
        dtype (type): The numpy dtype of the data in the file.
        queue_size (int, optional): Size of the queue for asynchronous batch generation. Defaults to 10.
        pad_token_id (int, optional): Token ID used for padding. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
    """
    def __init__(
            self, 
            token_ids_path: str, 
            max_seq_len: int, 
            batch_size: int, 
            dtype: type = np.uint16, 
            queue_size: int=10, 
            pad_token_id: int=0, 
            shuffle: bool=True) -> None:
        self.token_ids_path = token_ids_path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.queue_size = queue_size
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle

        # Open the numpy file in memory-map mode (read-only)
        self.token_ids = np.memmap(self.token_ids_path, dtype=self.dtype, mode='r')
        self.num_tokens = len(self.token_ids)
        self.total_possible_starts = self.num_tokens - max_seq_len

        if self.total_possible_starts <= 0:
            raise ValueError(
                f"Input array (length {self.num_tokens}) is too short to generate "
                f"input/target pairs of length {self.max_seq_len}. "
                f"Need at least {self.max_seq_len + 1} tokens."
            )

    def batch_generator(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        all_start_indices = np.arange(self.total_possible_starts)
        if self.shuffle:
            np.random.shuffle(all_start_indices)

        for i in range(0, self.total_possible_starts, self.batch_size):
            current_batch_start_token_indices = all_start_indices[i : i + self.batch_size]
            inputs_list = []
            targets_list = []

            for start_token_idx in current_batch_start_token_indices:
                end_inputs = start_token_idx + self.max_seq_len
                # Slicing the memmap array triggers the OS to read the data from disk
                inputs = self.token_ids[start_token_idx : end_inputs]
                targets = self.token_ids[start_token_idx + 1 : end_inputs + 1]
                inputs_list.append(inputs)
                targets_list.append(targets)

            # np.stack creates a new in-memory array from the slices, which is what we want for a batch.
            yield np.stack(inputs_list), np.stack(targets_list)

    def __len__(self) -> int:
        return (self.total_possible_starts + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return iter(self.__call__())

    def __del__(self):
        if hasattr(self, 'token_ids'):
            del self.token_ids

    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        data_queue = queue.Queue(maxsize=self.queue_size)
        def fill_queue():
            try:
                for batch in self.batch_generator():
                    data_queue.put(batch)
            except Exception as e:
                print(f"Error in data loading thread: {e}")
            finally:
                data_queue.put(None)

        threading.Thread(target=fill_queue, daemon=True).start()
        while True:
            batch = data_queue.get()
            if batch is None:
                break
            yield batch

def read_txt(file_path: os.PathLike | str) -> str:
    """Reads a single text file."""
    with open(file_path, 'r', encoding="utf-8") as f:
        text = f.read()
    return text

def read_all_text(directory_path: os.PathLike | str, sep: str = '\n' * 4) -> str:
    """Reads all text files in a given directory and concatenates them."""
    texts = []
    if not os.path.isdir(directory_path):
        print(f"Warning: Directory not found: {directory_path}. Returning empty string.")
        return ""
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.txt'):
            try:
                texts.append(read_txt(os.path.join(directory_path, file_name)))
            except (UnicodeDecodeError, PermissionError) as e:
                print(f"Skipping file {file_name} due to error: {e}")
    return sep.join(texts)

# Not really needed
def jax_np_collet(batch: Tuple[np.ndarray, np.ndarray], dtype=jnp.int32) -> Tuple[Array, Array]:
    input_array, target_array = batch
    return jnp.asarray(input_array, dtype=dtype), jnp.asarray(target_array, dtype=dtype)

def write_dataset_streaming(
    raw_text: str,
    tokenizer: MiniCharTok,
    out_path: str,
    dtype=np.uint16,
    chunk_chars: int = 1_000_000
) -> str:
    """
    Tokenizes a large text corpus in chunks and writes token IDs to a binary file.

    This streaming approach is memory-efficient as it avoids creating a single large
    list of all token IDs in memory. It processes the text piece by piece.

    Args:
        raw_text (str): The complete text data to be tokenized.
        tokenizer (MiniCharTok): An initialized character level tokenizer.
        out_path (str): The file path to save the binary output. '.bin' will be appended if not present.
        dtype (np.dtype, optional): The NumPy data type for storing token IDs.
            Defaults to np.uint16. Ensure this can hold the entire vocab size.
        chunk_chars (int, optional): The number of characters to process in each chunk.
            Defaults to 1,000,000.

    Returns:
        str: The final path to the created binary file.

    Raises:
        ValueError: If the tokenizer's vocabulary size is larger than what the
            specified `dtype` can represent.
    """
    if not out_path.endswith('.bin'):
        out_path += '.bin'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # If vocab won't fit in dtype, fail loudly
    vocab = tokenizer.GetPieceSize()
    max_id = vocab - 1
    max_dtype = np.iinfo(dtype).max
    if max_id > max_dtype:
        raise ValueError(f"Vocab {vocab} exceeds dtype {dtype} capacity ({max_dtype}). Use uint32.")

    with open(out_path, 'wb') as f:
        for i in range(0, len(raw_text), chunk_chars):
            chunk = raw_text[i:i+chunk_chars]
            ids = np.asarray(tokenizer.Encode(chunk), dtype=dtype)
            f.write(ids.tobytes())
    return out_path