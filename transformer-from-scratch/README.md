# Transformer Implementation: Deep Learning Systems Work Sample

This repository contains a from-scratch implementation of Transformer architectures, designed to demonstrate a deep understanding of sequence modeling, attention mechanisms, and end-to-end deep learning pipelines.

The codebase features a custom library, `lib`, which houses the core model definitions for both **Causal Language Models (CLM)** and **Automatic Speech Recognition (ASR)**. The accompanying notebook, `Train.ipynb`, serves as the entry point for training and evaluating the Causal Language Model.

## 🚀 Project Overview

The project is structured to showcase two main components:

1.  **`lib` (The Core Library):** A comprehensive deep learning library containing the "huge model" definitions. It supports:
    *   **Decoder-only Transformer:** For autoregressive tasks like text generation.
    *   **Encoder-Decoder Transformer:** For sequence-to-sequence tasks like ASR.
2.  **`Train.ipynb` (The Driver):** An interactive notebook that utilizes `lib` to train a Causal Language Model (CLM). It handles data loading, model instantiation, and the training loop.

## 📂 Key Components

### 1. `lib/` - The Model Backend
This directory contains the modular implementation of the Transformer architectures.

*   **`lib/model/`**: Contains the neural network modules.
    *   `transformers.py`: Defines the high-level `DecoderOnlyTransformer` and `EncoderDecoderTransformer` classes.
    *   `encoder_layers.py` / `decoder_layers.py`: Implements the specific transformer blocks.
    *   `sublayers.py`: Contains Feed-Forward Networks and Attention mechanisms.
    *   `positional_encoding.py`: Sinusoidal positional encodings.
    *   `speech_embedding.py`: Specialized embeddings for audio features (used in ASR).
*   **`lib/mytorch/`**: A custom implementation of core neural network layers (Linear, Activation, Attention) built from first principles to mirror PyTorch's internal logic.

### 2. `Train.ipynb` - The Training Workflow
Currently configured to run the **Causal Language Model (CLM)** task.

*   **Setup:** Initializes the environment and dependencies.
*   **Configuration:** Uses `config.yaml` to manage hyperparameters (layers, heads, dimensions, learning rates).
*   **Data Pipeline:** Loads text data using `lib.data` and prepares dataloaders for training, validation, and testing.
*   **Model Initialization:** Instantiates a `DecoderOnlyTransformer` from `lib`.
*   **Training Loop:** Executes the training process using the `LMTrainer`.

> **Note:** While `lib` includes full support for Automatic Speech Recognition (ASR) via the `EncoderDecoderTransformer` and `speech_embedding` modules, the current `Train.ipynb` is focused on the Language Modeling task.

## 🛠️ Technology Stack

*   **Python 3.12+**
*   **PyTorch:** Core deep learning framework.
*   **Custom "MyTorch":** Low-level implementation of attention and linear layers for educational transparency.
*   **WandB:** For experiment tracking and visualization.

## 💻 Usage

To replicate the training process:

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Notebook:**
    Open `Train.ipynb` and execute the cells. The notebook is self-contained and will:
    *   Load the configuration from `config.yaml`.
    *   Initialize the `DecoderOnlyTransformer`.
    *   Train the model on the provided dataset.

## 📊 Features Implemented

*   **Multi-Head Attention:** Scaled dot-product attention implemented from scratch.
*   **Positional Encodings:** Standard sinusoidal encodings for sequence order.
*   **Masking:** Causal masking for future-token hiding and padding masks for variable-length sequences.
*   **Beam Search & Greedy Decoding:** Logic for sequence generation.

---
*This repository serves as a work sample demonstrating the implementation of complex neural architectures and training pipelines without reliance on high-level abstraction libraries like HuggingFace Transformers for the core model logic.*
