# GPT Transformer Model

This repository contains a GPT-like transformer model built using PyTorch for natural language generation. The model is based on the architecture introduced in GPT-2, which has been trained on a custom dataset for text generation.

## Model Overview

The model is a multi-layer transformer-based neural network, consisting of the following components:

- **Causal Self-Attention:** A core component of the transformer that performs self-attention to process the input sequence.
- **MLP (Feedforward Layer):** Applied to each block in the transformer, which helps the model to learn complex relationships.
- **Layer Normalization:** Applied before each attention and feedforward layer to stabilize training.
- **Embedding Layers:** Token embeddings for words and positional embeddings for the sequence.

### Architecture
- **Embedding Dimension (`n_embd`)**: 768
- **Number of Attention Heads (`n_head`)**: 12
- **Number of Layers (`n_layer`)**: 12
- **Vocabulary Size (`vocab_size`)**: 50,257
- **Max Sequence Length (`block_size`)**: 1024

The model is trained for text generation and can be fine-tuned with custom data.

## Requirements

To run the model and perform inference, you will need the following dependencies:

- Python 3.7+
- PyTorch
- Gradio
- Transformers
- Tokenizers (GPT-2)
  
You can install the required libraries using:

```bash
pip install torch gradio transformers tiktoken
