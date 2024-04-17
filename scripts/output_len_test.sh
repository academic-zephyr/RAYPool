#!/bin/sh

conda activate utl
python baseline.py --device=0,1 --few-infer=0 --model ../../model/BAAI/AquilaChat2-34B --bitsandbytes 0 1>baseline.out 2>baseline.err
python main_vp.py --continuous-batch=0 --few-infer=0 --data-parallel-size 2 --model ../../model/AquilaChat2-GPTQ-34B-exlv2 1>vp.out 2>vp.err
python main_vp.py --continuous-batch=1 --few-infer=0 --data-parallel-size 2 --model ../../model/AquilaChat2-GPTQ-34B-exlv2 1>vp_cb.out 2>vp.err