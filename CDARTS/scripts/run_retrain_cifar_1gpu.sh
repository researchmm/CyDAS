NGPUS=1
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS ./CDARTS/retrain.py \
    --name cifar10-retrain --dataset cifar10 --model_type cifar \
    --n_classes 10 --init_channels 36 --stem_multiplier 3 \
    --cell_file './genotypes.json' \
    --batch_size 128 --workers 1 --print_freq 100 \
    --world_size $NGPUS --weight_decay 5e-4 \
    --distributed  --dist_url 'tcp://127.0.0.1:26443' \
    --lr 0.025 --warmup_epochs 0 --epochs 600 \
    --cutout_length 16 --aux_weight 0.4 --drop_path_prob 0.3 \
    --label_smooth 0.0 --mixup_alpha 0