# JAX Character-Level Transformer

This repository contains the implementation of a character-level Transformer model built with JAX and Flax (nnx). The model is designed to be trained on a text corpus and can generate new text, character by character. This project is inspired by Andrej Karpathy's [`nanoGPT`](https://github.com/karpathy/nanoGPT), [`minbpe`](https://github.com/karpathy/minbpe), and a bit by [`llm.c`](https://github.com/karpathy/llm.c) repositories, demonstrating a minimalist approach to building and training a Transformer model from scratch.

## Features

-   **Character-Level Tokenization**: A simple and effective tokenizer that works directly with characters.
-   **Modular Architecture**: The code is organized into modules for the model, data loading, training, and sampling, making it easy to understand and extend.
-   **JAX and Flax**: The model is implemented using JAX for high-performance numerical computing and Flax.nnx for neural network layers.
-   **Training and Sampling**: The repository includes a Jupyter notebook for training the model and generating new text.
-   **Educational**: The code is written to be clear and educational, trying to provide architectural insights into the inner workings of Transformers.

## Installation

To run this project, you'll need to have Python 3.x and the required packages installed.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Beiglar/Transformer_mini.git
    cd Transformer_mini
    ```

2.  **Install the dependencies:**

    It is recommended to use a virtual environment.

    ```bash
    pip install -r requirements.txt
    ```
    For installing JAX with GPU or TPU support, follow the instructions on the [JAX installation page](https://docs.jax.dev/en/latest/installation.html).

## Usage

The main entry point for this project is the `jax_char_transformer.ipynb` Jupyter notebook.

1.  **Open the notebook:**

    ```bash
    jupyter notebook jax_char_transformer.ipynb
    ```

2.  **Run the cells:**

    The notebook is divided into sections for:
    -   **Configuration**: Set hyperparameters for the model and training.
    -   **Data Preparation**: Preprocess the text data for training. Use your own data files in the `data/` directory.
    -   **Data Loading**: Load and tokenize the training data.
    -   **Model Definition**: Define the Transformer architecture.
    -   **Training**: Train the model on the text corpus.
    -   **Sampling**: Generate new text from the trained model.

    You can run the cells sequentially to train the model and see the generated output.

3. **Dive into `modular/` directory:**

   Explore the `modular/` directory to understand the implementation details of the model components.
   -   `config.py`: Contains configuration settings and hyperparameters for the model.
   -   `dataloader.py`: Handles data loading and preprocessing. The code is writen from ground up to be a performant memory-mapped multi-threaded data loader.
   -   `model.py`: Implements the Transformer model architecture using Flax.nnx.
   -   `nnx_modules.py`: Contains Flax.nnx implementations of Attention, RoPE, SwiGLU and transformer block modules used in the model.
   -   `sampling.py`: Implements sampling methods (top-p sampling and filtering) for text generation. These are used in main text sampling code that is a method of the `TinyTransformerLM` in `model.py`.
   -   `tokenizer.py`: Implements `MiniCharTok` class for tokenization and detokenization. You can swap this with sentencepiece for subword tokenization.
   -   `utils.py`: Contains utility functions for training and evaluation.
   -   `training.py`: Model optimizer and learning rate scheduler creation is handled in `create_model_and_optimizer` function, `train_step` is a jitted function for a single step of optimization and `compute_val_loss` does feedforward on model in eval mode and returns loss.

   You can use the code in this folder as baseline for your future JAX based transformer experimentation.

## Model Architecture

The model is a standard decoder-only Transformer with the following components:

-   **Token and Positional Embeddings**: Input characters are converted into embeddings, and positional information is added via Rotary Embeddings (rotates token embedding vectors based on their position in the sequence [RoPE paper here](https://arxiv.org/abs/2104.09864)).
-   **Multi-Head Self-Attention**: The core of the Transformer, allowing the model to weigh the importance of different characters in the input sequence.
-   **Feed-Forward Network**: GLU (Gated Linear Unit) applied after the attention mechanism.
-   **Layer Normalization**: Applied before the attention and feed-forward layers to stabilize training.

The entire model is implemented in the `modular/model.py` file.

## Future Work

-   Implement a more sophisticated tokenizer (e.g., BPE).
-   Add support for distributed training on multiple devices.
-   Experiment with different model architectures and hyperparameters.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
