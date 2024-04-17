from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import argparse
import json
import random
import time
from tqdm import tqdm
import os

"""Parse arguments"""

parser = argparse.ArgumentParser(description="Configure the inference")
parser.add_argument("--few-infer", type=int, default=0, help="Whether to run only few inference")
parser.add_argument("--device", type=str, default="0,1", help="The device to use for inference")
parser.add_argument("--model", type=str, default="../../model/BAAI/AquilaChat2-34B", help="Model to load")
parser.add_argument("--bitsandbytes", type=int, default=0, help="Whether to enable bits and bytes quantization")
args = parser.parse_args()

model_dir = args.model
seed = 0
dataset_dir = "../data/scrambled_sampled_dataset.json"

def assemble_prompt(prompt) -> str:
    """
    Assembles a prompt from a list of strings.

    Args:
        prompt (list): A list of strings.

    Returns:
        str: The assembled prompt.
    """
    return "USER: " + prompt + "\n" + "ASSISTANT:"


def inference(llm, tokenizer, requests):
    """
    Perform inference using a language model.

    Args:
        llm (LanguageModel): The language model used for inference.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input prompts.
        requests (list): A list of inference requests, where each request is a tuple
            containing the prompt, a placeholder value, and the desired output length.

    Returns:
        tuple: A tuple containing the total inference time (in seconds) and the total
            number of tokens processed during inference.

    """
    input_num_tokens = []
    output_num_tokens = []
    start_time = time.perf_counter()
    
    for request in tqdm(requests):
        prompt, _, output_len = request
        input_ids = tokenizer(assemble_prompt(prompt), return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=False,
            num_return_sequences=1,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=output_len,
        )
        # Include the decoding time.
        tokenizer.decode(llm_outputs[0], skip_special_tokens=True)
        input_num_tokens.append(len(input_ids[0]))
        output_num_tokens.append(len(llm_outputs[0]))
    
    end_time = time.perf_counter()
    return end_time - start_time, sum(output_num_tokens)
    

if __name__=="__main__":

    """Configure the environment"""

    random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tensor_parallel_size = len(args.device.split(","))

    """Load the model and tokenizer"""

    quantization_config=BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
    llm = AutoModelForCausalLM.from_pretrained(model_dir, 
                                               trust_remote_code=True, 
                                               torch_dtype=torch.bfloat16,
                                               device_map="balanced",
                                               quantization_config=None
                                               if args.bitsandbytes == 0 
                                               else quantization_config,)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                              trust_remote_code=True)

    """Load the dataset"""

    with open(dataset_dir) as f:
        requests = json.load(f)
        
    # Few infer: run requests with output_length < 5 only. This is mean to quickly test the performance.
    if args.few_infer == 1:
        requests = [request for request in requests if request[2] in (24, 48, 62)]
        
    """Start inference"""
    elapsed_time, total_num_tokens = inference(llm=llm, tokenizer=tokenizer, requests=requests)
        
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s")
    print(f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s")
    print(f"Total_num_tokens: {total_num_tokens:.2f} tokens")
    print(f"Total_time: {elapsed_time:.2f} s")
