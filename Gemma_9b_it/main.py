import argparse
import requests
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
import torch
import os
import ast
from huggingface_hub import login


login(token="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")

# Constants for inference only
NUTRITIONIX_API_KEY = "09a7857598b3b41a3fc083e7410107d7"
NUTRITIONIX_APP_ID = "5c7c84ad"
NUTRITIONIX_API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

def load_dataset_from_csv(csv_path):
    """Load dataset from CSV file"""
    df = pd.read_csv(csv_path)
    df = df[['title', 'ingredients', 'directions']]  
    dataset = Dataset.from_pandas(df)
    return dataset

def preprocess_training_data(examples, tokenizer):
    """Process training data into input-output format"""
    processed_data = []
    
    for i in range(len(examples["title"])):
        ingredients = ast.literal_eval(examples["ingredients"][i]) if isinstance(examples["ingredients"][i], str) else examples["ingredients"][i]
        directions = ast.literal_eval(examples["directions"][i]) if isinstance(examples["directions"][i], str) else examples["directions"][i]
        
        # Format: Input is just title, Output includes ingredients and directions
        formatted_text = f"""Input: Generate a recipe for {examples["title"][i]}
Output: Here's the recipe for {examples["title"][i]}:

Ingredients:
{', '.join(ingredients)}

Directions:
{' '.join(directions)}"""
        
        processed_data.append(formatted_text)
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        processed_data,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = tokenized["input_ids"].clone()
    return tokenized

def train_model(train_csv, output_dir):
    print("Initializing training...")

    torch.cuda.empty_cache()
    
    # Load the tokenizer and model for google/gemm-2-9b-it
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    tokenizer.padding_side = 'right'
    
    train_dataset = load_dataset_from_csv(train_csv)
    tokenized_train = train_dataset.map(
        lambda x: preprocess_training_data(x, tokenizer),
        batched=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        load_in_8bit=True, 
    )

    print("Model named modules:")
    for name, module in model.named_modules():
        print(name)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj"],  # Update these based on the model architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable parameter: {name}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="no",
        fp16=True,  # Enable FP16 if the model supports it
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        learning_rate=5e-4,  # Adjust learning rate as needed
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit", 
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        max_seq_length=256,
        tokenizer=tokenizer
    )

    trainer.train()
    save_path = os.path.join(output_dir, "saved_model")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Training completed and model saved.")

def generate_recipe(recipe_name, model, tokenizer):
    """Generate recipe and add nutrition info (inference only)"""
    input_text = f"Input: Generate a recipe for {recipe_name}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True 
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['train', 'inference'],
        help="Whether to train model or run inference"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        help="Path to training CSV file (required for training)"
    )
    parser.add_argument(
        "--recipe_name",
        type=str,
        help="Recipe name for inference"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./out_dir",
        help="Directory for saving/loading model"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.train_csv:
            raise ValueError("--train_csv required for training mode")
        train_model(args.train_csv, args.model_dir)
    
    elif args.mode == 'inference':
        if not args.recipe_name:
            raise ValueError("--recipe_name required for inference mode")
        
        model_path = os.path.join(args.model_dir, "saved_model")
        if not os.path.exists(model_path):
            raise ValueError(f"No saved model found at {model_path}")
        
        print("Loading saved model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            load_in_8bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        recipe_output = generate_recipe(args.recipe_name, model, tokenizer)
        print("\nGenerated Recipe:\n")
        print(recipe_output)

if __name__ == "__main__":
    main()
