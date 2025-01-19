print("Step 1: Importing libraries...")
import torch
from transformers import GPT2Tokenizer
import gradio as gr
from Assign12_Model import GPT, GPTConfig
import torchvision
torchvision.disable_beta_transforms_warning()

print("Step 2: Loading the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)
model = GPT(config)

print("Step 3: Loading model weights...")
model.load_state_dict(torch.load("trained_model.pth", map_location=device, weights_only=True))
model.eval().to(device)

print("Step 4: Loading tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add print statements in the function
print("Step 5: Defining the inference function...")
def generate_text(prompt, max_length=50, num_return_sequences=1):
    print(f"Received input prompt: {prompt}")
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = []
    for _ in range(num_return_sequences):
        with torch.no_grad():
            logits, _ = model(inputs)
            generated_token = torch.argmax(logits[:, -1, :], dim=-1)
            inputs = torch.cat((inputs, generated_token.unsqueeze(0)), dim=1)
            if inputs.size(1) >= max_length:
                break
        output = tokenizer.decode(inputs[0].tolist())
        outputs.append(output)
    return outputs

import os
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import gradio as gr

# Initialize model and tokenizer
model_name = 'gpt2'  # You can replace this with your specific model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()

# Ensure we're using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define parameters
max_length = 100
num_return_sequences = 1  # Default number of sequences to generate

# Function to generate text
def generate_text(prompt, max_len=50, num_outputs=1):
    global max_length, num_return_sequences
    
    max_length = max_len
    num_return_sequences = num_outputs
    
    # Encode the input text
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    generated_sequences = []  # List to store generated text

    # Generate sequences
    with torch.no_grad():
        for i in range(num_return_sequences):
            x = input_ids.clone()
            while x.size(1) < max_length:
                logits = model(x).logits  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1)  # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                x = torch.cat((x, xcol), dim=1)

            # Decode the generated tokens and append it to the list
            tokens = x[0, :max_length].tolist()
            decoded = tokenizer.decode(tokens, skip_special_tokens=True)
            generated_sequences.append(f"Generated Text {i+1}:")
            generated_sequences.append(f"> {decoded}\n")

    # Join the generated sequences into a structured output
    structured_output = "\n".join(generated_sequences)
    
    return structured_output

# Set up Gradio interface
print("Step 6: Setting up the Gradio interface...")
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Input Prompt"),
        gr.Slider(10, 200, step=10, label="Max Length", value=50),
        gr.Number(label="Number of Outputs", value=10),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Transformer Text Generator",
    description="Enter a prompt and generate text using the trained transformer model.",
)

print("Step 7: Launching the Gradio interface...")
interface.launch(share=True)
