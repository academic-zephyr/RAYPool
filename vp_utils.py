import socket
import ray
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import numpy as np

@ray.remote(num_cpus=1, num_gpus=1)
class VLLMRemote:
    def __init__(self, model):
        self.llm = LLM(model=model,
                       trust_remote_code=True,
                       quantization="gptq",
                       tensor_parallel_size=1,)
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.input_num_tokens = []  # record the number of total input tokens
        self.output_num_tokens = [] # record the number of total output tokens
        
    def generate(self, input_ids, sampling_params):
        """Generate the output of the model given the input_ids and sampling_params."""
        llm_outputs = self.llm.generate(prompt_token_ids=input_ids, 
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
        self.input_num_tokens.append(len(input_ids[0]))
        self.output_num_tokens.append(len(llm_outputs[0].outputs[0].token_ids))
        return llm_outputs
    
    def batched_generate(self, input_ids, sampling_params):
        """Batched version of generate."""
        llm_outputs = self.llm.generate(prompt_token_ids=input_ids, 
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
        for i in range(len(llm_outputs)):
            self.input_num_tokens.append(len(input_ids[i]))
            self.output_num_tokens.append(len(llm_outputs[i].outputs[0].token_ids))
        return llm_outputs        
    
    def host(self):
        """Return the hostname of the machine where the remote object is initialized."""
        return socket.gethostbyname(socket.gethostname())
    
    def clear_num_tokens(self):
        """Clear the num_tokens list."""
        self.input_num_tokens = []
        self.output_num_tokens = []
        return "num_tokens has been cleaned"
    
    def get_num_tokens(self):
        """Return the total number of tokens processed by the model."""
        return sum(self.input_num_tokens) + sum(self.output_num_tokens)
    
    
    """Utilities for evaluating accuracy"""
    def eval_generate(self, input_ids):
        sampling_params = SamplingParams(logprobs=100008, max_tokens=1)
        llm_outputs = self.llm.generate(prompt_token_ids=input_ids, 
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
        logits = llm_outputs[0].outputs[0].logprobs[-1]
        
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[self.tokenizer("A").input_ids[-1]],
                        logits[self.tokenizer("B").input_ids[-1]],
                        logits[self.tokenizer("C").input_ids[-1]],
                        logits[self.tokenizer("D").input_ids[-1]],
                    ]
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        
        return pred
    
    def batched_eval_generate(self, input_ids):
        preds = []
        sampling_params = SamplingParams(logprobs=100008, max_tokens=1)
        llm_outputs = self.llm.generate(prompt_token_ids=input_ids, 
                                        sampling_params=sampling_params,
                                        use_tqdm=False)
        
        for i in range(len(llm_outputs)):
            logits = llm_outputs[i].outputs[0].logprobs[-1]
            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(
                        [
                            logits[self.tokenizer("A").input_ids[-1]],
                            logits[self.tokenizer("B").input_ids[-1]],
                            logits[self.tokenizer("C").input_ids[-1]],
                            logits[self.tokenizer("D").input_ids[-1]],
                        ]
                    ).float(),
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
            preds.append(pred)
            
        return preds
        
        
        