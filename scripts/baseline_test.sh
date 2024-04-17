#!/bin/sh

conda activate utl
python baseline.py --device=0,1 --few-infer=0 --model ../../model/BAAI/AquilaChat2-34B --bitsandbytes 0 1>baseline.out 2>baseline.err
python baseline.py --device=0,1 --few-infer=0 --model ../../model/BAAI/AquilaChat2-34B --bitsandbytes 1 1>baseline_bab.out 2>baseline_bab.err