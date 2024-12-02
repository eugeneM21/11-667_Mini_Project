import argparse
import requests
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import torch
import os
import math
import ast
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


login(token="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")

NUTRITIONIX_API_KEY = "09a7857598b3b41a3fc083e7410107d7"
NUTRITIONIX_APP_ID = "5c7c84ad"
NUTRITIONIX_API_URL = "https://trackapi.nutritionix.com/v2/natural/nutrients"

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
tokenizer.padding_side = 'right'


def extract_input(prompt_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "Prompts", prompt_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file not found at {file_path}")

    with open(file_path, "r") as file:
        input_text = file.read()
    return input_text


def load_dataset_from_csv(csv_path):
    """
    Load a dataset from a CSV file and convert it to a Hugging Face Dataset.
    """
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    return dataset


def fetch_nutritional_info(ingredient):
    """
    Fetches nutritional information for a given ingredient from the Nutritionix API.
    """
    headers = {
        "x-app-id": NUTRITIONIX_APP_ID,
        "x-app-key": NUTRITIONIX_API_KEY,
        "Content-Type": "application/json"
    }
    data = {"query": ingredient}
    response = requests.post(NUTRITIONIX_API_URL, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to fetch data for {ingredient}"}


def get_nutritional_data(ingredients):
    """
    Fetch nutritional data for a list of ingredients.
    """
    nutrients_info = []
    for ingredient in ingredients.split(", "):  # Split ingredients by commas
        nutrition_data = fetch_nutritional_info(ingredient)
        if "foods" in nutrition_data:
            for food in nutrition_data["foods"]:
                nutrients_info.append({
                    "ingredient": food.get("food_name", ingredient),
                    "calories": food.get("nf_calories", "N/A"),
                    "protein": food.get("nf_protein", "N/A"),
                    "carbs": food.get("nf_total_carbohydrate", "N/A"),
                    "fats": food.get("nf_total_fat", "N/A")
                })
    return nutrients_info


def format_nutritional_output(nutrients_info):
    """
    Formats the nutritional data into a readable string.
    """
    nutrient_text = "\nNutritional Details:\n"
    for nutrient in nutrients_info:
        nutrient_text += (
            f"Ingredient: {nutrient['ingredient']}, "
            f"Calories: {nutrient['calories']} kcal, "
            f"Protein: {nutrient['protein']} g, "
            f"Carbs: {nutrient['carbs']} g, "
            f"Fats: {nutrient['fats']} g\n"
        )
    return nutrient_text


def format_str(examples):
    prepend_title = "This recipe is: "
    prepend_ingredients = "\nThe list of ingredients are: "
    prepend_directions = "\nThe directions to make this recipe is: "
    prepend_NER = "\nThe NER for this recipe is: "
    processed_data = []

    for i in range(len(examples["title"])):
        new_str = prepend_title + examples["title"][i]

        # Safely parse the string fields into lists
        ingredients = ast.literal_eval(examples["ingredients"][i]) if isinstance(examples["ingredients"][i], str) else examples["ingredients"][i]
        ingredients = " ".join(ingredients)
        new_str += prepend_ingredients + ingredients

        directions = ast.literal_eval(examples["directions"][i]) if isinstance(examples["directions"][i], str) else examples["directions"][i]
        directions = " ".join(directions)
        new_str += prepend_directions + directions

        ner = ast.literal_eval(examples["NER"][i]) if isinstance(examples["NER"][i], str) else examples["NER"][i]
        ner = " ".join(ner)
        new_str += prepend_NER + ner

        processed_data.append(new_str)

    return processed_data


def preprocess_data(examples):
    """
    Tokenizes the formatted examples for training.
    """
    formatted_examples = format_str(examples)
    
    tokenized_inputs = tokenizer(
        formatted_examples,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
    
    return tokenized_inputs


def inference_with_nutrients(input_text, model, tokenizer):
    """
    Performs inference and dynamically adds nutritional information.
    """
    tokenized_input = tokenizer(input_text, return_tensors="pt").to("cuda")
    generation = model.generate(**tokenized_input, max_new_tokens=512)
    raw_output = tokenizer.decode(generation[0], skip_special_tokens=True)

    start_idx = input_text.find("The list of ingredients are: ") + len("The list of ingredients are: ")
    end_idx = input_text.find("\n", start_idx)
    ingredients = input_text[start_idx:end_idx].strip()

    nutrients_info = get_nutritional_data(ingredients)
    nutrient_text = format_nutritional_output(nutrients_info)

    final_output = raw_output + nutrient_text

    return final_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="A txt file specifying prompt. Just provide name of prompt file. Not full path."
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="./train/train.csv",
        help="Path to the training dataset CSV file."
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default="./out_dir/saved_model",
        help="Path to the saved model directory."
    )
    args = parser.parse_args()

    if os.path.exists(args.saved_model_dir):
        print("Saved model found. Loading the model...")
        model = AutoModelForCausalLM.from_pretrained(args.saved_model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.saved_model_dir)
    else:
        print("No saved model found. Starting training...")
        train_dataset = load_dataset_from_csv(args.train_csv)

        tokenized_train = train_dataset.map(preprocess_data, batched=True)

        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-9b-it",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )

        lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.1,
        bias="none",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
        output_dir="out_dir",
        logging_steps=1,
        num_train_epochs=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        learning_rate=0.000005,
        weight_decay=0.005,
        warmup_ratio=0.006,
        lr_scheduler_type="cosine"
        )
        trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        max_seq_length=256,
        tokenizer=tokenizer
        )

        trainer.train()

        model.save_pretrained(args.saved_model_dir)
        tokenizer.save_pretrained(args.saved_model_dir)
        print("Model training completed and saved.")


    input_text = extract_input(args.prompt)
    output = inference_with_nutrients(input_text, model, tokenizer)
    print(output)



if __name__ == "__main__":
    main()
