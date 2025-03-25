export PYTHONPATH=SGM_train
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

conda activate sgm_train

cd SGM_train

python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --accum_iter 4 \
    --model mae_vit_base_patch16_dec512d2b \
    --token_size 14 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --log_dir work_dirs/debug \
    --output_dir work_dirs/debug \
    LOGGING.eval_interval 50 \
    DATASET.root mapdata/gibson \
    DATASET.enable_unexp_area True
