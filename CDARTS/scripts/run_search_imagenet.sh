NGPUS=8
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS ./CDARTS/search.py \
    --name imagenet-search --dataset imagenet --model_type imagenet \
    --n_classes 1000 --init_channels 16 --stem_multiplier 1 \
    --distributed  --world_size $NGPUS --dist_url 'tcp://127.0.0.1:23343' \
    --batch_size 128 --sample_ratio 0.2 \
    --workers 4 --print_freq 10 \
    --use_apex --sync_param \
    --regular --regular_ratio 0.5 --regular_coeff 5 \
    --clean_arch --loss_alpha 1 \
    --ensemble_param \
    --w_lr 0.8 --alpha_lr 3e-4 --nasnet_lr 0.8 \
    --w_weight_decay 1e-5 --alpha_weight_decay 1e-5 \
    --one_stage --repeat_cell \
    --interactive_type 0 \
    --pretrain_epochs 10 --pretrain_decay 3 \
    --search_iter 40 --search_iter_epochs 1 --nasnet_warmup 3