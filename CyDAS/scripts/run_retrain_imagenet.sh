NGPUS=8
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS ./CDARTS/retrain.py \
    --name imagenet-retrain --dataset imagenet --model_type imagenet \
    --n_classes 1000 --init_channels 48 --stem_multiplier 1 \
    --batch_size 128 --workers 4 --print_freq 100 \
    --cell_file './genotypes.json' \
    --world_size $NGPUS --weight_decay 3e-5 \
    --distributed  --dist_url 'tcp://127.0.0.1:24443' \
    --lr 0.5 --warmup_epochs 5 --epochs 250 \
    --cutout_length 0 --aux_weight 0.4 --drop_path_prob 0.0 \
    --label_smooth 0.1 --mixup_alpha 0 \
    --resume_name "retrain_resume.pth.tar"