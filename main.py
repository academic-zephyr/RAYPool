from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import json
import random
import time
import os

"""Parse arguments"""

parser = argparse.ArgumentParser(description="Configure the inference")
parser.add_argument("--few-infer", type=int, default=0, help="Whether to run only few inference")
parser.add_argument("--continuous-batch", type=int, default=0, help="Whether to activate continuous batch")
parser.add_argument("--device", type=str, default="0,1", help="The device to use for inference")
parser.add_argument("--model", type=str, default="../../model/AquilaChat2-GPTQ-34B-exlv1", help="Model to load")
parser.add_argument("--multi-node", type=int, default=0, help="Whether to run on multi-node")
args = parser.parse_args()

quantized_model_dir = args.model
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
    

def group_data(requests) -> dict:
    """
    Groups the requests based on their output length.

    Args:
        requests (list): A list of requests, where each request is a tuple of three elements:
                         (input_data, target_data, output_len).

    Returns:
        dict: A dictionary where the keys are the output lengths and the values are lists of requests
              with the corresponding output length.
    """
    # Group the batch by output length
    requests_by_output_len = {}
    for request in requests:
        _, _, output_len = request
        if output_len not in requests_by_output_len:
            requests_by_output_len[output_len] = []
        requests_by_output_len[output_len].append(request)

    return requests_by_output_len


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
    intput_num_tokens = []
    output_num_tokens = []
    start_time = time.perf_counter()
    
    for request in tqdm(requests):
        prompt, _, output_len = request
        input_ids = [tokenizer(assemble_prompt(prompt)).input_ids]
        sampling_params = SamplingParams(n=1, best_of=1, temperature=0, top_p=1.0, top_k=50, max_tokens=output_len)
        llm_outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params, use_tqdm=False)
        intput_num_tokens.append(len(input_ids[0]))
        output_num_tokens.append(len(llm_outputs[0].outputs[0].token_ids))
    
    end_time = time.perf_counter()
    return end_time - start_time, sum(intput_num_tokens) + sum(output_num_tokens)
    

def batched_inference(llm, tokenizer, requests_by_output_len):
    """
    Perform batched inference using a language model.

    Args:
        llm (LanguageModel): The language model used for inference.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input prompts.
        requests_by_output_len (dict): A dictionary mapping output lengths to a list of requests.

    Returns:
        tuple: A tuple containing the total time taken for inference and the total number of tokens processed.

    """
    intput_num_tokens = []
    output_num_tokens = []
    start_time = time.perf_counter()
    
    for output_len, requests in tqdm(requests_by_output_len.items()):
        sampling_params = SamplingParams(n=1, best_of=1, temperature=0, top_p=1.0, top_k=50, max_tokens=output_len)
        prompts = [assemble_prompt(request[0]) for request in requests]
        input_ids = tokenizer(prompts, padding=False).input_ids
        print(f"Output_len: {output_len}, #Prompt: {len(input_ids)}")
        llm_outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params, use_tqdm=False)
        for i in range(len(llm_outputs)):
            intput_num_tokens.append(len(input_ids[i]))
            output_num_tokens.append(len(llm_outputs[i].outputs[0].token_ids))
    
    end_time = time.perf_counter()
    return end_time - start_time, sum(intput_num_tokens) + sum(output_num_tokens)
    

if __name__=="__main__":

    """Configure the environment"""

    random.seed(seed)
    if args.multi_node == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        tensor_parallel_size = len(args.device.split(","))
    else:
        tensor_parallel_size = 4

    """Load the model and tokenizer"""

    llm = LLM(model=quantized_model_dir,
              trust_remote_code=True,
              quantization="gptq",
              tensor_parallel_size=tensor_parallel_size,
              enforce_eager=True if args.multi_node == 1 else False,)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                              trust_remote_code=True)

    """Load the dataset"""

    with open(dataset_dir) as f:
        requests = json.load(f)
        
    # Few infer: run requests with output_length < 5 only. This is mean to quickly test the performance.
    if args.few_infer == 1:
        requests = [request for request in requests if request[2] in (24, 48, 62)]
        
    """Start inference"""
    if args.continuous_batch == 1:
        print("Continuous batch is activated.")
        requests_by_output_len = group_data(requests=requests)
        elapsed_time, total_num_tokens = batched_inference(llm=llm, tokenizer=tokenizer, requests_by_output_len=requests_by_output_len)
    else:
        elapsed_time, total_num_tokens = inference(llm=llm, tokenizer=tokenizer, requests=requests)
        
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s")
    print(f"Tokens/s: {total_num_tokens / elapsed_time:.2f} tokens/s")
    print(f"Total_num_tokens: {total_num_tokens:.2f} tokens")
    print(f"Total_time: {elapsed_time:.2f} s")
