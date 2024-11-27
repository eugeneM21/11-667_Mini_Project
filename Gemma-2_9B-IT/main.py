import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
tokenizer.padding_side = 'right'

def extract_input(prompt_file):
   file_path = "../Prompts/" + prompt_file
   file = open(file_path, "r")
   input_text = file.read()
   return input_text

# Formatting the training dataset to feed to finetuned model
def format_str(examples):
    prepend_title = "This recipe is: "
    prepend_ingredients = "\nThe list of ingredients are: "
    prepend_directions = "\nThe directions to make this recipe is: "
    prepend_NER = "\nThe NER for this recipe is: "
    processed_data = []
    for i in range(len(examples["title"])):
        new_str = prepend_title + examples["title"][i]
        ingredients = examples["ingredients"][i]
        ingredients = " ".join(ingredients)
        new_str += prepend_ingredients + ingredients
        direction = examples["directions"][i]
        direction  = " ".join(direction)
        new_str += prepend_directions + direction
        ner = examples["ner"][i]
        ner  = " ".join(ner)
        new_str += prepend_NER + ner

        print(new_str)
        print(type(new_str))
        exit()
        processed_data.append(new_str)
    print(processed_data)
    print(len(processed_data))
    return processed_data


def preprocess_data(examples):
    formatted_examples = format_str(examples)
    return tokenizer(formatted_examples, padding="max_length", truncation=True)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
       "--prompt",
       type=str,
       required=True,
       help="A txt file specifying prompt. Just provide name of prompt file. Not full path."
    )
    #parser.add_argument(
    #   "--output",
    #   type=str,
    #   rquired=True,
    #   help="A jsonl file with LLM generation"
    #)
    args = parser.parse_args()


    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )

    # Load train and val dataset
    train_dataset= load_dataset("recipe_nlg", split='train[0:2000]', data_dir="../dataset")
    val_dataset= load_dataset("recipe_nlg", split='train[2000:2500]', data_dir="../dataset")
    


    #print(train_dataset.map(preprocess_data, batched=True))
    # Preprocess dataset for finetuning
    tokenized_train = train_dataset.map(preprocess_data, batched=True)

    # Set up LORA
    lora_config = LoraConfig(
       r=16,
       lora_alpha=8,
       target_modules=["q_proj","k_proj"],
       lora_dropout=0.1,
       bias="none",
       #modules_to_save=["classifier"]
    )
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)

    # Set up training arguments
    training_args = TrainingArguments(
    output_dir = "out_dir",
    #eval_strategy = "epoch",
    logging_steps = 1,
    num_train_epochs= 1,
    save_strategy = "no",
    bf16 = True,
    gradient_checkpointing = True,
    per_device_train_batch_size = 8,
    learning_rate = 0.000005,
    weight_decay = 0.005,
    warmup_ratio = 0.006,
    lr_scheduler_type = "cosine"
    )

    # Finetune
    trainer = SFTTrainer(
            model=model,
            args = training_args,
            train_dataset=tokenized_train,
            max_seq_length=64,
            tokenizer=tokenizer
            )
    trainer.train()


    # Read prompt from input txt file
    input_text = extract_input(args.prompt)

    # Testing inference
    tokenized_input = tokenizer(input_text, return_tensors="pt").to("cuda")
    generation = model.generate(**tokenized_input, max_new_tokens=512)
    print(tokenizer.decode(generation[0]))



if __name__ == "__main__":
  main()