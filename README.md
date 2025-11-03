# LLMs from Scratch

A TensorFlow implementation of GPT-style language models built from the ground up, demonstrating core transformer architecture components and training workflows.

## Project Overview

This project implements a custom GPT model using TensorFlow/Keras, covering everything from text tokenization to self-attention mechanisms. The implementation focuses on educational clarity while maintaining functional architecture patterns used in modern language models.

## Features

- **Custom Tokenization Pipeline**: Regex-based tokenizer with dictionary mapping for text-to-token conversion
- **GPT-2 Tokenizer Integration**: Byte Pair Encoding (BPE) using tiktoken library for subword tokenization
- **Windowing Data Loader**: Custom TensorFlow data pipeline with configurable stride and context length
- **Token & Positional Embeddings**: Combined embedding layers for semantic and positional information
- **Causal Self-Attention**: Implementation of masked self-attention mechanism preventing future token access
- **Training on Shakespeare Dataset**: Uses the Tiny Shakespeare dataset (1M+ characters) for model training

## Implementation Details

### Data Processing
- Dataset: Tiny Shakespeare (1,003,854 characters)
- Tokenization: GPT-2 BPE tokenizer (50,257 vocabulary size)
- Context window: 100 tokens
- Batch size: 8
- Stride: 4 (for efficient sequence sampling)

### Model Architecture
- Embedding dimension: 256
- Positional encoding: Learned positional embeddings
- Attention mechanism: Causal (masked) self-attention with query, key, value projections
- Masking: Upper triangular mask to prevent attending to future tokens

## Notebooks

**basic_gpt.ipynb**: Complete walkthrough of building a GPT model from scratch, including:
  - Text preprocessing and tokenization
  - Custom DataLoader implementation
  - Embedding layer construction
  - Self-attention mechanism (both simple and causal variants)
  - Training setup with TensorFlow/Keras

## Requirements

- tensorflow>=2.15.0
- tensorflow-datasets
- tiktoken
- numpy
- matplotlib
- plotly



