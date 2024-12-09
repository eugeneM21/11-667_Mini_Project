import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import os
import ast
from huggingface_hub import login
from evaluate import load
from sentence_transformers import SentenceTransformer, util


# Login to HuggingFace
login(token="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")

def calculate_instructions_bert(model, tokenizer, dataset_path, max_length=512):
    """
    Calculates perplexity for the model's generated output.
    """
    val_df = pd.read_csv(dataset_path)
    
    if 'title' not in val_df.columns:
        raise ValueError("The dataset must contain 'title' column.")
    
    #num_samples = len(val_df)
    bert_precision = 0
    bert_recall = 0
    bert_f1 = 0
    ingredient_match_score = 0
    avg_similarity_score = 0
    num_samples = 50

    
    for idx, row in val_df.iterrows():
        title = row['title']
        target_instructions = [' '.join(ast.literal_eval(row['directions']))]
        target_ingredients = ast.literal_eval(row['NER'])
        print(target_ingredients)
        similarity_model = SentenceTransformer('paraphrase-MiniLM-L12-v2')
        
        bertscore = load('bertscore')
        ingredient_match_curr = 0

        
        input_text = f"Input: Generate a detailed recipe for {title}, including ingredients and cooking instructions.\nOutput:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        #for gemma model validation
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

        #for gemma model validation
        if "token_type_ids" in generated_inputs:
            del generated_inputs["token_type_ids"]
        
        if "Directions" in generated_text:
            generated_instructions = [generated_text.split("Directions",1)[1].strip()]
        elif "Instructions" in generated_text: 
            generated_instructions = [generated_text.split("Instructions",1)[1].strip()]
        elif "instructions" in generated_text: 
            generated_instructions = [generated_text.split("instructions",1)[1].strip()]

        
        bert_results = bertscore.compute(predictions=generated_instructions, references=target_instructions, lang="en")
        
        bert_precision += bert_results['precision'][0]
        bert_recall += bert_results['recall'][0]
        bert_f1 += bert_results['f1'][0]

        
        generated_embedding = similarity_model.encode(generated_instructions[0], convert_to_tensor=True)
        target_embedding = similarity_model.encode(target_instructions[0], convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(generated_embedding, target_embedding)[0][0]
        avg_similarity_score += similarity_score

        for ingredient in target_ingredients:
            if ingredient in generated_text:
                ingredient_match_curr += 1
        
        ingredient_match_curr = ingredient_match_curr/len(target_ingredients)
        ingredient_match_score += ingredient_match_curr

        print(f"Bert Score for row {idx + 1}: {bert_results}\n")
        print(f"Ingredient Match for row {idx + 1}: {ingredient_match_curr}\n")
        print(f"Directions Similarity score for row {idx + 1}: {similarity_score}\n")


        if idx+1 >= num_samples:
            break

    bert_precision = bert_precision/num_samples
    bert_recall = bert_recall/num_samples
    bert_f1 = bert_f1/num_samples
    ingredient_match_score = ingredient_match_score/num_samples
    avg_similarity_score = avg_similarity_score/num_samples

    
    return bert_precision, bert_recall, bert_f1, ingredient_match_score, avg_similarity_score


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
        
        input_text = f"Input: Generate a recipe for {title}.\nOutput:"
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

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
        "--model_name",
        type=str,
        default="google/gemma-2-9b-it",
        help="Name of the pretrained model on Hugging Face Hub"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for input-output"
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="Metric to calculate. Type perplexity or bert"
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.metric == "perplexity":
        print("Calculating perplexity for the model...")
        perplexity = calculate_perplexity(model, tokenizer, args.val_csv, max_length=args.max_length)
        print(f"Validation Perplexity: {perplexity}")
    else:
        print("Calculating Bert score for the model...")
        bert_precision, bert_recall, bert_f1, ingredient_match_score, similarity_score = calculate_instructions_bert(model, tokenizer, args.val_csv, max_length=args.max_length)
        print(f"Bert Score Precision: {bert_precision}, Recall: {bert_recall}, F1: {bert_f1}")
        print(f"Ingredient match_score: {ingredient_match_score}")
        print(f"Directions Similarity score: {similarity_score}")

if __name__ == "__main__":
    main()
