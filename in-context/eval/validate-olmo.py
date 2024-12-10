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
    Calculates perplexity for the model's generated in-context.
    """
    val_df = pd.read_csv(dataset_path)

    if 'title' not in val_df.columns:
        raise ValueError("The dataset must contain 'title' column.")

    total_loss = 0
    num_samples = len(val_df)
    total_tokens = 0
    results = []  # List to store each row's results

    for idx, row in val_df.iterrows():
        title = row['title']
        reference_text = row['ingredients'] + row['directions']

        input_text = f"Input: Generate a detailed recipe for {title}, including ingredients and cooking instructions.\nOutput:"
        # input_text = (f"Create a recipe for the dish {title}. Include the [ingredients], and [directions]. \n"
        #               f"Here is an example of a recipe:\\n\\[title]: No-Bake Nut Cookies\\n [ingredients]: [\"1 c. firmly packed brown sugar\", \"1/2 c. evaporated milk\", \"1/2 tsp. vanilla\", \"1/2 c. broken nuts (pecans)\", \"2 Tbsp. butter or margarine\", \"3 1/2 c. bite size shredded rice biscuits\"]\\n [directions]: [\"In a heavy 2-quart saucepan, mix brown sugar, nuts, evaporated milk and butter or margarine.\", \"Stir over medium heat until mixture bubbles all over top.\", \"Boil and stir 5 minutes more. Take off heat.\", \"Stir in vanilla and cereal; mix well.\", \"Using 2 teaspoons, drop and shape into 30 clusters on wax paper.\", \"Let stand until firm, about 30 minutes.\"]")

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length,
                           return_token_type_ids=False)
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

            total_tokens += new_tokens

        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
        if not generated_text:
            print(f"Generated text for row {idx + 1} is empty.")
            continue

        print(f"Generated Recipe for row {idx + 1}:\n{generated_text}\n")

        generated_inputs = tokenizer(generated_text, return_tensors="pt", truncation=True, max_length=max_length,
                                     return_token_type_ids=False)
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
                "reference_text": reference_text,
                "generated_text": generated_text,
                "perplexity": row_perplexity
            })

    avg_loss = total_loss / num_samples
    overall_perplexity = math.exp(avg_loss)

    avg_loss2 = total_loss / total_tokens
    overall_perplexity2 = math.exp(avg_loss2)

    results_df = pd.DataFrame(results)
    results_df.to_csv("output_results-olmo-zero-shot.csv", index=False)
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
        help="Maximum sequence length for input-in-context"
    )
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    print(val_df)

    # model_path = os.path.join(args.model_dir, "saved_model")
    # if not os.path.exists(model_path):
    #     raise ValueError(f"No saved model found at {model_path}")

    model_path = "allenai/OLMo-7B-Instruct"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Calculating perplexity for the model...")
    ppl1, ppl2 = calculate_perplexity(model, tokenizer, args.val_csv, max_length=args.max_length)
    print(f"Validation Perplexity: {ppl1}, {ppl2}")


if __name__ == "__main__":
    main()
