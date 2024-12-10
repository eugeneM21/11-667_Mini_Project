import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import os
from huggingface_hub import login
from vllm import LLM, SamplingParams
import torch
import numpy as np

# Load the model with vLLM
llm = LLM(model="allenai/OLMo-7B-Instruct", trust_remote_code=True)  # Replace "gpt2" with your model's name (e.g., "gpt3", "bloom", etc.)

# Function to compute perplexity
def compute_perplexity(text, llm):
    # Tokenize the input text
    tokens = llm.tokenizer.encode(text, return_tensors="pt")
    num_tokens = tokens.size(1)

    # Compute log probabilities
    sampling_params = SamplingParams(output_token_scores=True)
    output = llm.generate(text, sampling_params=sampling_params)

    # Extract token log probabilities
    token_scores = output.token_scores  # Log probabilities of each token
    log_probs = torch.tensor(token_scores)

    # Calculate perplexity
    avg_log_prob = torch.mean(log_probs)  # Average log probability
    perplexity = torch.exp(-avg_log_prob).item()  # Exponentiate the negative mean log probability

    return perplexity

# Example text
text = "Generate a recipe for carrot cake"

# Compute perplexity
perplexity = compute_perplexity(text, llm)

print(f"Perplexity: {perplexity}")