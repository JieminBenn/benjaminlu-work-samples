# Transformer from Scratch

This project implements Transformer architectures from scratch as part of my exploration of sequence modeling and deep learning systems. The goal was to understand and build the core components — attention, masking, positional encoding, and training logic — without relying on high-level abstraction libraries.

> **Note**: This project was originally inspired by academic coursework but has been refactored into an independent implementation for portfolio demonstration.

## Features

- **Decoder-Only Transformer**: Suitable for language modeling tasks (GPT-style).
- **Encoder-Decoder Transformer**: Implementation available for sequence-to-sequence tasks.
- **Custom Attention Layers**: Explicit implementation of Scaled Dot-Product and Multi-Head Attention.
- **Training Pipeline**: modular trainer with logging, checkpointing, and visualization.

## Project Structure

```
├── lib/
│   ├── model/          # Transformer components (Layers, Embedding, Positional Encoding)
│   ├── data/           # Dataset and Tokenizer handling
│   ├── trainers/       # Training loop and experiment management
│   └── utils/          # Utilities for optimization and scheduling
├── data/               # Data storage
├── Train.ipynb         # Main entry point for training and experimentation
├── config.yaml         # Configuration for the model and training
└── requirements.txt    # Python dependencies
```

## Setup & Usage

1. **Environment Setup**
   It is recommended to use a virtual environment (conda or venv).

   ```bash
   conda create -n transformer python=3.10
   conda activate transformer
   pip install -r requirements.txt
   ```

2. **Configuration**
   The model and training parameters are defined in `config.yaml`. You can modify model size, batch size, learning rates, etc.

3. **Running the Code**
   The primary interface is the Jupyter Notebook `Train.ipynb`. 
   
   - Launch Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open `Train.ipynb` and run the cells to initialize the model and start training.

## License

This project is for educational and portfolio purposes.
