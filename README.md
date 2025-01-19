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
    
    python app.py
    
    Access the web interface from your browser and input prompts to generate text.
### Model Training Logs
    During training, the loss is printed after each epoch, and once the target loss is reached, the model will be saved for best_loss in 100 epoch iterations. 
    Here’s an example of the output logs from training:
```
using device: cuda
Number of parameters: 124,439,808
loaded 338025 tokens
1 epoch = 165 batches
Epoch 1/100: 100%|████████████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=6]
Epoch 2/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=5.48]
Epoch 3/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.74it/s, loss=5.14]
Epoch 4/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.98]
Epoch 5/100: 100%|██████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.8]
Epoch 6/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.62]
Epoch 7/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.57]
Epoch 8/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.44]
Epoch 9/100: 100%|█████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.32]
Epoch 10/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.29]
Epoch 11/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.16]
Epoch 12/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=4.03]
Epoch 13/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.91]
Epoch 14/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.79]
Epoch 15/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.68]
Epoch 16/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.57]
Epoch 17/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.39]
Epoch 18/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.34]
Epoch 19/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.29]
Epoch 20/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.74it/s, loss=3.14]
Epoch 21/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=3.05]
Epoch 22/100: 100%|████████████████████████████████████| 165/165 [01:34<00:00,  1.75it/s, loss=2.96]
Epoch 23/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.71it/s, loss=2.91]
Epoch 24/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.70it/s, loss=2.76]
Epoch 25/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.71it/s, loss=2.64]
Epoch 26/100: 100%|████████████████████████████████████| 165/165 [01:35<00:00,  1.72it/s, loss=2.62]
Epoch 27/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.70it/s, loss=2.52]
Epoch 28/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.72it/s, loss=2.42]
Epoch 29/100: 100%|████████████████████████████████████| 165/165 [01:38<00:00,  1.67it/s, loss=2.32]
Epoch 30/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.68it/s, loss=2.25]
Epoch 31/100: 100%|████████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=2.15]
Epoch 32/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.70it/s, loss=2.09]
Epoch 33/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.69it/s, loss=2.01]
Epoch 34/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.72it/s, loss=1.92]
Epoch 35/100: 100%|████████████████████████████████████| 165/165 [01:36<00:00,  1.71it/s, loss=1.86]
Epoch 36/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.70it/s, loss=1.75]
Epoch 37/100: 100%|████████████████████████████████████| 165/165 [01:38<00:00,  1.68it/s, loss=1.67]
Epoch 38/100: 100%|████████████████████████████████████| 165/165 [01:38<00:00,  1.67it/s, loss=1.61]
Epoch 39/100: 100%|████████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=1.53]
Epoch 40/100: 100%|████████████████████████████████████| 165/165 [01:35<00:00,  1.72it/s, loss=1.43]
Epoch 41/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.70it/s, loss=1.37]
Epoch 42/100: 100%|████████████████████████████████████| 165/165 [01:37<00:00,  1.70it/s, loss=1.27]
Epoch 43/100: 100%|████████████████████████████████████| 165/165 [01:40<00:00,  1.65it/s, loss=1.22]
Epoch 44/100: 100%|████████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=1.13]
Epoch 45/100: 100%|████████████████████████████████████| 165/165 [01:39<00:00,  1.67it/s, loss=1.01]
Epoch 46/100: 100%|███████████████████████████████████| 165/165 [01:42<00:00,  1.60it/s, loss=0.934]
Epoch 47/100: 100%|███████████████████████████████████| 165/165 [01:37<00:00,  1.68it/s, loss=0.901]
Epoch 48/100: 100%|████████████████████████████████████| 165/165 [01:38<00:00,  1.67it/s, loss=0.79]
Epoch 49/100: 100%|███████████████████████████████████| 165/165 [01:41<00:00,  1.63it/s, loss=0.723]
Epoch 50/100: 100%|███████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=0.637]
Epoch 51/100: 100%|███████████████████████████████████| 165/165 [01:39<00:00,  1.66it/s, loss=0.582]
Epoch 52/100: 100%|███████████████████████████████████| 165/165 [01:41<00:00,  1.63it/s, loss=0.493]
Epoch 53/100: 100%|███████████████████████████████████| 165/165 [01:41<00:00,  1.63it/s, loss=0.434]
Epoch 54/100: 100%|███████████████████████████████████| 165/165 [01:37<00:00,  1.70it/s, loss=0.365]
Epoch 55/100: 100%|███████████████████████████████████| 165/165 [01:38<00:00,  1.67it/s, loss=0.306]
```

