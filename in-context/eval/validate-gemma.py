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
    total_tokens =0
    results = []  # List to store each row's results

    for idx, row in val_df.iterrows():
        title = row['title']
        reference_text = row['ingredients'] + row['directions']

        #input_text = f"Input: Generate a detailed recipe for {title}, including ingredients and cooking instructions.\nOutput:"
        # input_text = (f"Create a recipe for the dish {title}. Include the [ingredients], and [directions]. \n"
        #               f"Here is an example of a recipe:\\n\\[title]: No-Bake Nut Cookies\\n [ingredients]: [\"1 c. firmly packed brown sugar\", \"1/2 c. evaporated milk\", \"1/2 tsp. vanilla\", \"1/2 c. broken nuts (pecans)\", \"2 Tbsp. butter or margarine\", \"3 1/2 c. bite size shredded rice biscuits\"]\\n [directions]: [\"In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.\", \"Stir over medium heat until mixture bubbles all over top.\", \"Boil and stir 5 minutes more. Take off heat.\", \"Stir in vanilla and cereal; mix well.\", \"Using 2 teaspoons, drop and shape into 30 clusters on wax paper.\", \"Let stand until firm, about 30 minutes.\"]")

        input_text = f"""
        Generate a recipe for {title} only providing a list of ingredients, and step-by-step directions. 
        Here are examples below. Do not add them to the output:
        Example 1:
        Dish: No-Bake Nut Cookies
        Ingredients:
        - 1 c. firmly packed brown sugar
        - 1/2 c. evaporated milk
        - 1/2 tsp. vanilla
        - 1/2 c. broken nuts (pecans)
        - 2 Tbsp. butter or margarine
        - 3 1/2 c. bite-size shredded rice biscuits

        Directions:
        1. In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk, and butter or margarine.
        2. Stir over medium heat until the mixture bubbles all over the top.
        3. Boil and stir for 5 minutes more. Remove from heat.
        4. Stir in vanilla and cereal; mix well.
        5. Using 2 teaspoons, drop and shape into 30 clusters on wax paper.
        6. Let stand until firm, about 30 minutes.

        Example 2:
        Dish: Jewell Ball's Chicken
        Ingredients:
        - 1 small jar chipped beef, cut up
        - 4 boned chicken breasts
        - 1 can cream of mushroom soup
        - 1 carton sour cream

        Directions:
        1. Place chipped beef on the bottom of a baking dish.
        2. Place chicken on top of the beef.
        3. Mix the soup and cream together; pour over the chicken.
        4. Bake, uncovered, at 275°F for 3 hours.

        Example 3:
        Dish: Creamy Corn
        Ingredients:
        - 2 (16 oz.) pkg. frozen corn
        - 1 (8 oz.) pkg. cream cheese, cubed
        - 1/3 c. butter, cubed
        - 1/2 tsp. garlic powder
        - 1/2 tsp. salt
        - 1/4 tsp. pepper

        Directions:
        1. In a slow cooker, combine all ingredients.
        2. Cover and cook on low for 4 hours or until heated through and the cheese is melted.
        3. Stir well before serving.

        """

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length, return_token_type_ids=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

            # `generated_tokens` contains the token IDs
            new_tokens = generated_tokens.shape[1]  # Total tokens per sequence
            print(f"Total tokens generated per sequence: {new_tokens}")

            total_tokens+= new_tokens
        
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        if not generated_text:
            print(f"Generated text for row {idx + 1} is empty.")
            continue
        
        print(f"Generated Recipe for row {idx + 1}:\n{generated_text}\n")
        
        generated_inputs = tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=max_length,  return_token_type_ids=False)
        labels = generated_inputs.input_ids.clone()
        generated_inputs = {k: v.to(model.device) for k, v in generated_inputs.items()}
        labels = labels.to(model.device)
        
        with torch.no_grad():
            outputs = model(**generated_inputs, labels=labels)
            loss = outputs.loss.item()
            row_perplexity = math.exp(loss)
            print(f"Perplexity for row {idx + 1}: {row_perplexity}\n")
            total_loss += loss


            # Add row results to list
            results.append({
                "title": title,
                "input_text": input_text,
                "generated_text": generated_text,
                "perplexity": row_perplexity
            })

    avg_loss = total_loss / num_samples
    overall_perplexity = math.exp(avg_loss)

    avg_loss2 = total_loss / total_tokens
    overall_perplexity2 = math.exp(avg_loss2)

    results_df = pd.DataFrame(results)
    results_df.to_csv("output_results-gemma.csv", index=False)
    print(f"Results saved to output_csv")

    return overall_perplexity, overall_perplexity2

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
        required=False,
        help="Directory containing the saved model (Gemma or Olmo)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for input-output"
    )
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    print(val_df)

    # model_path = os.path.join(args.model_dir, "saved_model")
    # if not os.path.exists(model_path):
    #     raise ValueError(f"No saved model found at {model_path}")

    model_path = "google/gemma-2-9b-it"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Calculating perplexity for the model...")
    ppl1,ppl2 = calculate_perplexity(model, tokenizer, args.val_csv, max_length=args.max_length)
    print(f"Validation Perplexity: {ppl1}, {ppl2}")

if __name__ == "__main__":
    main()
