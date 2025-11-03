# LLMs from Scratch

A complete TensorFlow implementation of a GPT-style transformer language model built from the ground up. This project demonstrates every component of modern transformer architecture, from tokenization to multi-head attention and full model training.

## Project Overview

This project implements a 124M parameter GPT model using TensorFlow/Keras, replicating the architecture used in GPT-2. The implementation covers the full pipeline from text preprocessing to a trainable transformer model with causal self-attention.

## Features

### Tokenization
- **Custom Regex Tokenizer**: Simple dictionary-based tokenizer with special tokens (`<|unk|>`, `<|eos|>`)
- **GPT-2 BPE Tokenizer**: Production-grade Byte Pair Encoding using tiktoken library (50,257 vocabulary)

### Data Pipeline
- **Custom TensorFlow DataLoader**: Windowing-based data loader with configurable stride and context length
- **Efficient Batching**: Prefetching and shuffling support for optimized training
- **Input-Target Pairing**: Automatic creation of training pairs using sliding window technique

### Model Architecture
- **Token & Positional Embeddings**: Combined 768-dimensional embeddings with learned positional encoding
- **Multi-Head Causal Attention**: Full implementation with query-key-value projections and masking
- **Transformer Blocks**: Complete transformer decoder stack with residual connections
- **Layer Normalization**: Custom implementation with learnable scale and shift parameters
- **GELU Activation**: Gaussian Error Linear Unit activation function
- **Feed-Forward Networks**: 4x expansion ratio FFN with GELU activation
- **Dropout Regularization**: Configurable dropout on embeddings, attention, and FFN layers

## Architecture Details

### GPT-124M Configuration

{
"vocab_size": 50257, # GPT-2 tokenizer vocabulary
"context_length": 1024, # Maximum sequence length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of transformer blocks
"drop_rate": 0.1, # Dropout probability
"qkv_bias": False # Query-Key-Value projection bias
}


**Total Parameters**: ~124 million trainable parameters[file:42]

### Attention Mechanism

The model implements three levels of attention complexity:

1. **Simple Attention**: Basic self-attention with softmax normalization
2. **Causal Self-Attention**: Single-head masked attention preventing future token access
3. **Multi-Head Attention**: Parallel attention heads with concatenation and projection

Attention formula: \( \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \)[file:42]

### Training Configuration
- **Dataset**: Tiny Shakespeare (1,003,854 characters)
- **Context window**: 1024 tokens
- **Batch size**: 8
- **Stride**: 8 (overlapping sequences for data augmentation)

## Implementation Classes

### Core Components

**SimpleTockenizer**: Dictionary-based tokenizer with encode/decode methods[file:42]

**DataLoader**: Custom TensorFlow data pipeline with windowing support[file:42]

**SelfAttention**: Basic self-attention layer with Q, K, V projections[file:42]

**CasualSelfAttention**: Single-head attention with causal masking and dropout[file:42]

**MultiHeadAttention**: Production-grade multi-head attention with reshaping and masking[file:42]

**TransformerBlock**: Complete transformer decoder block with layer norm and residual connections[file:42]

**LayerNorm**: Custom layer normalization with learnable parameters[file:42]

**GELU**: Gaussian Error Linear Unit activation function[file:42]

**FeedForward**: Position-wise feed-forward network with 4x expansion[file:42]

**GPTModel**: Complete GPT model with embedding layers, transformer stack, and output head[file:42]

## Requirements

 - tensorflow>=2.15.0
 - tensorflow-datasets
 - tiktoken
 - numpy
 - matplotlib
