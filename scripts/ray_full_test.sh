#!/bin/sh

conda activate vllm
python main.py --continuous-batch=1 --multi-node=1 --few-infer=0 --model ../../model/AquilaChat2-GPTQ-34B-exlv2 1>ray_cb.out 2>ray_cb.err
python main.py --continuous-batch=0 --multi-node=1 --few-infer=0 --model ../../model/AquilaChat2-GPTQ-34B-exlv2 1>ray.out 2>ray.err