#!/bin/bash

export PYTHONPATH=..SGM
export CUDA_VISIBLE_DEVICES=0,1
cd ..SGM/semexp

# conda activate SGM

python eval_sgm.py \
  --split val \
  --seed 345 \
  --eval 1 \
  --pf_model_path ..pretrained_models/MAE-checkpoint-199.pth \
  -d ..experiments \
  --num_local_steps 1 \
  --exp_name "debug" \
  --global_downscaling 1 \
  --mask_nearest_locations \
  --pf_masking_opt 'unexplored' \
  --use_nearest_frontier \
  --total_num_scenes "5" \
  --step_test 5 \
  --num_area 9 \
  --thr 0.35 \
  --mask_num 49 \
  --expand_ratio 0.1 \
  --print_images "1" \
  --num_pf_maps "3"
