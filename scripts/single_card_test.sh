#!/bin/sh

conda activate vllm
python main_vp.py --continuous-batch=0 --few-infer=1 --data-parallel-size 1 --model ../../model/AquilaChat2-GPTQ-34B-exlv2
python main_vp.py --continuous-batch=1 --few-infer=1 --data-parallel-size 1 --model ../../model/AquilaChat2-GPTQ-34B-exlv2
