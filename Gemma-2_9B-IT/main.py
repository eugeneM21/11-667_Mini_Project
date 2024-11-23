import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

def extract_input(prompt_file):
   file_path = "../Prompts/" + prompt_file
   file = open(file_path, "r")
   input_text = file.read()
   return input_text

def preprocess_data(examples):
    examples["element"] = "The recipe/dish is: " + examples["element"]["title"] + "\nThe list of ingredients are: "  + "\nThe directions to make this are: " + "\n"
    return examples
    #print(examples["directions"])
    #print(''.join(examples["directions"]))
    #exit()
    #return "The recipe/dish is: " + examples["title"] + "\nThe list of ingredients are: " + ''.join(examples["ingredients"]) + "\nThe directions to make this are: " + ''.join(examples["directions"]) + "\n"

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

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load train and val dataset
    train_dataset= load_dataset("recipe_nlg", split='train[0:2000]', data_dir="../dataset")
    val_dataset= load_dataset("recipe_nlg", split='train[2000:2500]', data_dir="../dataset")
    #val_dataset = load_dataset("recipe_nlg", split='test[2000:2500]', data_dir="../dataset")
    #train_dataset = dataset["train"]
    print(train_dataset[0])
    print(train_dataset.shape)
    print(type(train_dataset))
    print(val_dataset[0])
    print(val_dataset.shape)
    print(type(val_dataset))
    #val_dataset = dataset["train"]
    print(train_dataset[0]["directions"])
    print(''.join(train_dataset[0]["directions"]))
    print(type(''.join(train_dataset[0]["directions"])))

    #print(train_dataset.map(preprocess_data, batched=True))

    trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            max_seq_length=512,
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