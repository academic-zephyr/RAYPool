from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer
from multiprocessing.pool import ThreadPool
from multiprocessing import Queue, Manager
from categories import categories, subcategories
from vp_utils import VLLMRemote
import numpy as np
import pandas as pd
import argparse
import os
import json
import random
import math
import ray

"""Parse arguments"""

parser = argparse.ArgumentParser(description="Configure the inference")
parser.add_argument("--ntrain", "-k", type=int, default=5)
parser.add_argument("--save-dir", "-s", type=str, default="./eval_results")
parser.add_argument("--data-dir", "-d", type=str, default="data")
parser.add_argument("--continuous-batch", type=int, default=0, help="Whether to activate continuous batch")
parser.add_argument("--model", type=str, default="../../model/AquilaChat2-34B-GPTQ-exlv2", help="Model to load")
parser.add_argument("--data-parallel-size", type=int, default=1, help="Data parallelism size")
args = parser.parse_args()

quantized_model_dir = args.model
seed = 0
choices = ['A', 'B', 'C', 'D']

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt
    

def group_data_idx(iter_length, llm_num):    
    if iter_length <= llm_num:
        return [range(iter_length)]
    else:
        batch_size = math.ceil(iter_length / llm_num)
        num_batches = math.ceil(iter_length / batch_size)
        grouped_idx = [range(i*batch_size,(i+1)*batch_size) for i in range(num_batches-1)]
        grouped_idx.append(range((num_batches-1)*batch_size, iter_length))
        return grouped_idx

def inference_worker(q, input_ids, label, cors):
    llm_id = q.get()
    pred = ray.get(llm_id.eval_generate.remote(input_ids))
    q.put(llm_id)
    
    cor = pred == label
    cors.append(cor)
    

def batched_inference_worker(q, input_ids_list, label_list, cors):
    llm_id = q.get()
    preds = ray.get(llm_id.batched_eval_generate.remote(input_ids_list))
    q.put(llm_id)
    
    for i in range(len(preds)):
        cor = preds[i] == label_list[i]
        cors.append(cor)

def inference(args, subject, llm_ids, tokenizer, dev_df, test_df):
    manager = Manager()
    cors = manager.list([])
    
    """Prepare ThreadPool"""
    
    q = Queue()
    for llm_id in llm_ids:
        q.put(llm_id)
        
    pool = ThreadPool(len(llm_ids))
    
    for i in range(test_df.shape[0]):
        """Assemble prompt"""
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        input_ids = [tokenizer(prompt).input_ids]
        
        """Check if the prompt is too long"""
        while len(input_ids[0]) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = [tokenizer(prompt).input_ids]
            
        """Get the label"""
        
        label = test_df.iloc[i, test_df.shape[1] - 1]
        
        """Appedn to ThreadPool"""
    
        while True:
            if not q.empty():
                pool.apply_async(inference_worker, args=(q, input_ids, label, cors))
                break
        
    pool.close()
    pool.join()
        
    acc = np.mean(cors)
    cors = np.array(cors)
    
    print(f"Average accuracy {acc:.3f} - {subject}")
    return cors, acc
    

def batched_inference(args, subject, llm_ids, tokenizer, dev_df, test_df):
    manager = Manager()
    cors = manager.list([])
    
    """Prepare ThreadPool"""
    
    q = Queue()
    for llm_id in llm_ids:
        q.put(llm_id)
        
    pool = ThreadPool(len(llm_ids))
    
    group_idx = group_data_idx(test_df.shape[0], len(llm_ids))
    
    for i in range(len(group_idx)):
        """Assemble prompt"""
        input_ids_list = []
        for idx in group_idx[i]:
            k = args.ntrain
            prompt_end = format_example(test_df, idx, include_answer=False)
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            
            input_ids = [tokenizer(prompt).input_ids]
            
            """Check if the prompt is too long"""
            while len(input_ids[0]) > 2048:
                k -= 1
                train_prompt = gen_prompt(dev_df, subject, k)
                prompt = train_prompt + prompt_end
                input_ids = [tokenizer(prompt).input_ids]
            
            input_ids_list.append(input_ids[0])
        
        """Get the label"""
        
        label_list = []
        for idx in group_idx[i]:
            label = test_df.iloc[idx, test_df.shape[1] - 1]
            label_list.append(label)
        
        """Appedn to ThreadPool"""
    
        while True:
            if not q.empty():
                pool.apply_async(batched_inference_worker, args=(q, input_ids_list, label_list, cors))
                break
        
    pool.close()
    pool.join()
        
    acc = np.mean(cors)
    cors = np.array(cors)
    
    print(f"Average accuracy {acc:.3f} - {subject}")
    return cors, acc
    

if __name__=="__main__":

    """Configure the environment"""

    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    ray.init()
    random.seed(seed)

    """Load the model and tokenizer"""

    llm_ids = [VLLMRemote.remote(quantized_model_dir) for _ in range(args.data_parallel_size)]
    
    hosts = ray.get([llm_id.host.remote() for llm_id in llm_ids])
    for host in hosts:
        print("Model is ready at", host)

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                              trust_remote_code=True)

    """Load the dataset"""

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )
    
    """Specify subjects"""
    
    all_cors = []  # All True False lists for different categories
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    
    """Start inference"""
    
    for subject in tqdm(subjects):
        dev_df = pd.read_csv(
            os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None
        )[: args.ntrain]
        test_df = pd.read_csv(
            os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None
        )
        
        if args.continuous_batch == 0:
            cors, acc = inference(args, subject, llm_ids, tokenizer, dev_df, test_df)
        else:
            cors, acc = batched_inference(args, subject, llm_ids, tokenizer, dev_df, test_df)
            
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)
        
    """Summary"""
    print("########### Summary #############")
            
    results = {"subcategories": {}, "categories": {}}
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))
        
    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
        
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))
    
    """Log results"""
    
    results_file = os.path.join(
        args.save_dir, "accuracies.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f)
