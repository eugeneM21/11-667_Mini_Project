import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import os
from huggingface_hub import login

# Login to HuggingFace
login(token="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")

def calculate_perplexity(model, tokenizer, dataset_path, max_length=512):
    """
    Calculates perplexity for the model's generated output.
    """
    val_df = pd.read_csv(dataset_path)
    
    if 'title' not in val_df.columns:
        raise ValueError("The dataset must contain 'title' column.")
    
    total_loss = 0
    num_samples = len(val_df)
    
    for idx, row in val_df.iterrows():
        title = row['title']
        
        input_text = f"Input: Generate a detailed recipe for {title}, including ingredients and cooking instructions.\nOutput:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #for olmo model validation
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        if not generated_text:
            print(f"Generated text for row {idx + 1} is empty.")
            continue
        
        print(f"Generated Recipe for row {idx + 1}:\n{generated_text}\n")
        
        generated_inputs = tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=max_length)
        labels = generated_inputs.input_ids.clone()
        generated_inputs = {k: v.to(model.device) for k, v in generated_inputs.items()}
        labels = labels.to(model.device)

        #for olmo model validation
        if "token_type_ids" in generated_inputs:
            del generated_inputs["token_type_ids"]
        
        with torch.no_grad():
            outputs = model(**generated_inputs, labels=labels)
            loss = outputs.loss.item()
            row_perplexity = math.exp(loss)
            print(f"Perplexity for row {idx + 1}: {row_perplexity}\n")
            total_loss += loss
    
    avg_loss = total_loss / num_samples
    overall_perplexity = math.exp(avg_loss)
    
    return overall_perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the saved model (Gemma or Olmo)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for input-output"
    )
    args = parser.parse_args()
    
    model_path = os.path.join(args.model_dir, "saved_model")
    if not os.path.exists(model_path):
        raise ValueError(f"No saved model found at {model_path}")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Calculating perplexity for the model...")
    perplexity = calculate_perplexity(model, tokenizer, args.val_csv, max_length=args.max_length)
    print(f"Validation Perplexity: {perplexity}")

if __name__ == "__main__":
    main()
