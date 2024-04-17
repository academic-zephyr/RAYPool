#!/bin/sh

conda activate utl
# python eval_vp.py --data-dir ../tools/MMLU_eval/data/ \
#                   --save-dir ../log/eval/bs1 \
#                   --continuous-batch 0 \
#                   --data-parallel-size 4
python eval_vp.py --data-dir ../tools/MMLU_eval/data/ \
                  --save-dir ../log/eval/cb \
                  --continuous-batch 1 \
                  --data-parallel-size 6