NGPUS=1
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS ./CDARTS/search.py \
    --name cifar10-search --dataset cifar10 --model_type cifar \
    --n_classes 10 --init_channels 16 --layer_num 3 --stem_multiplier 3 \
    --batch_size 64 --sample_ratio 1.0 \
    --workers 1 --print_freq 10 \
    --distributed --world_size $NGPUS --dist_url 'tcp://127.0.0.1:23343' \
    --use_apex --sync_param \
    --regular --regular_ratio 0.667 --regular_coeff 5 \
    --clean_arch --loss_alpha 1 \
    --ensemble_param \
    --w_lr 0.08 --alpha_lr 3e-4 --nasnet_lr 0.08 \
    --w_weight_decay 3e-4 --alpha_weight_decay 0. \
    --one_stage --repeat_cell --fix_head \
    --interactive_type 3 \
    --pretrain_epochs 2 --pretrain_decay 0 \
    --search_iter 32 --search_iter_epochs 1 --nasnet_warmup 1