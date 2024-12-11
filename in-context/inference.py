"""
This is an example code. Feel free to modify it when needed.
"""

import argparse
import json
from typing import List, Union, Dict

import requests
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def determine_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

MODEL_MAP = {
    "pretrained": "allenai/OLMo-7B-hf",
    "sft": "allenai/OLMo-7B-SFT-hf",
    "instruct": "allenai/OLMo-7B-Instruct-hf",
    "gemma":"google/gemma-2-9b-it"
}


def convert_prompt_to_input(data: []) -> List[Union[str, Dict[str, str]]]:
    return [{"role": "user", "content": item} for item in data]


@torch.inference_mode()
def generate_output_list(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        messages: List[Union[str, Dict[str, str]]],
        model_type: str,
        device: str
) -> []:
    outputs=[]
    for message in tqdm(messages):
        prompt = message if isinstance(message, str) else message["content"]

        if model_type == "pretrained":
            inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        else:
            inputs = tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

        if model_type != "pretrained" and not isinstance(inputs, dict):
            inputs = {'input_ids': inputs}
        inputs = {k: v.to(device) for k, v in inputs.items()}

        response = model.generate(**inputs, max_new_tokens=300)
        response_decoded = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

        if model_type != "pretrained":
            response_decoded = response_decoded.split("\n<|assistant|>\n")[-1]

        outputs.append(response_decoded)

    return outputs


def load_model_and_generate_output(data: []) -> []:
    messages = convert_prompt_to_input(data)
    print(f'messages: {messages}')

    device = determine_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP['instruct'],
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP['instruct'])


    return generate_output_list(model, tokenizer, messages, 'instruct', device)

def hit_vllm_model_and_generate_output(data: []) -> []:
    messages = convert_prompt_to_input(data)
    #print(f'messages: {messages}')

    # Define the server URL
    url = "http://localhost:8000/v1/chat/completions"
    # Prepare headers for the request
    headers = {
        "Content-Type": "application/json"
    }

    outputs = []
    for message in tqdm(messages):
        # Prepare the data payload with the current content
        data = {
            "model": "allenai/OLMo-7B-Instruct-hf",
            "messages": [
               message
            ]
        }


        # Make the POST request
        response = requests.post(url, headers=headers, data=json.dumps(data))

        #print(f'response: {response} \n\n')
        print(f'data: {data} \n\n')

        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()  # Parse the JSON response
            #print(f'result: {result}')
            # Extract the 'content' value

            content = result['choices'][0]['message']['content']
            #print(f'content: {content}\n\n')

            outputs.append(content)  # Add the result to the outputs list

        else:
            print(f"Failed to get response for content: '{message}'")
            print("Status Code:", response.status_code)
            print("Response:", response.text)

        #break


    return outputs


def prepare_messages(file_path: str, model_type: str) -> List[Union[str, Dict[str, str]]]:
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]
    
    if model_type == "pretrained":
        return [item["prompt"] for item in data]
    else:
        return [{"role": "user", "content": item["prompt"]} for item in data]


@torch.inference_mode()
def generate_output_to_file(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Union[str, Dict[str, str]]],
    output_file: str,
    model_type: str,
    device: str
):
    with open(output_file, "w") as f:
        for message in tqdm(messages):
            prompt = message if isinstance(message, str) else message["content"]
            
            if model_type == "pretrained":
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
            else:
                inputs = tokenizer.apply_chat_template([message], tokenize=True, add_generation_prompt=True, return_tensors="pt")

            if model_type != "pretrained" and not isinstance(inputs, dict):
                inputs = {'input_ids': inputs}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = model.generate(**inputs, max_new_tokens=1000)
            response_decoded = tokenizer.batch_decode(response, skip_special_tokens=True)[0]

            if model_type != "pretrained":
                response_decoded = response_decoded.split("\n<|assistant|>\n")[-1]

            json.dump({"prompt": prompt, "output-1": response_decoded}, f)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_MAP.keys(),
        help="The type of the model: pretrained, sft, or instruct"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=True,
        help="A jsonl file where each line has an input prompt"
    )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     required=True,
    #     help="A jsonl file where each line will have the input and the corresponding output-1"
    # )
    args = parser.parse_args()

    device = determine_device()

    # print(f'model-{MODEL_MAP[args.model_type]}')
    messages = prepare_messages(args.prompts, args.model_type)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model_type], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_type])

    generate_output_to_file(model, tokenizer, messages, args.output, args.model_type, device)

    print("Done!")

if __name__ == "__main__":
    main()