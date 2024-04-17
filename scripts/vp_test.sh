#!/bin/sh

conda activate vllm
python main_vp.py --continuous-batch=1 --few-infer=0 --data-parallel-size 6 --model ../../model/AquilaChat2-34B-GPTQ-exlv2