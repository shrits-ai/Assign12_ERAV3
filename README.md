# GPT Transformer Model

This repository contains a GPT-like transformer model built using PyTorch for natural language generation. The model is based on the architecture introduced in GPT-2, which has been trained on a custom dataset (`input.txt` ) for text generation.

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
```
### Training the Model
## Steps for Training:
    Data Preparation: The training data is read from a text file (input.txt) and tokenized using tiktoken.
    Model Configuration: The GPTConfig class defines hyperparameters such as the sequence length, embedding size, number of layers, and more.
    Training Loop: The training loop uses the AdamW optimizer and a batch size of 16 with a sequence length of 128.
    Loss Tracking: The model trains until the target loss is reached, and then the model's weights are saved to a file (trained_model.pth).

### Inference with the Trained Model
    Once the model is trained, you can load the weights and generate text based on a given prompt using the provided app.py. 
    

  # Gradio Interface
    The model is accessible via a simple web interface built using Gradio. 
    Users can input a prompt, set the desired maximum length for the generated text, and specify how many text sequences to generate. 
    The model will return the generated text.

    -Input Prompt: A text input box where users can enter a prompt for text generation.
    -Max Length: A slider to adjust the maximum length of the generated text.
    -Number of Outputs: A number input to control how many different generated texts to return.
    The output will be displayed below the input controls as generated text.

  #  Running the Web Interface
    Clone the repository and install the required dependencies.
    Ensure that you have the trained model weights (trained_model.pth) in the same directory as app.py.
    Run the app.py file to start the Gradio web interface
    ```
    python app.py
    ```
    Access the web interface from your browser and input prompts to generate text.
### Model Training Logs
    During training, the loss is printed after each epoch, and once the target loss is reached, the model will be saved for best_loss in 100 epoch iterations. 
    Here’s an example of the output logs from training:

