NGPUS=4
SGPU=0
EGPU=$[NGPUS+SGPU-1]
GPU_ID=`seq -s , $SGPU $EGPU`
CUDA_VISIBLE_DEVICES=$GPU_ID python -m torch.distributed.launch --nproc_per_node=$NGPUS ./CDARTS/test.py \
    --name imagenet-retrain --dataset imagenet --model_type imagenet \
    --n_classes 1000 --init_channels 48 --stem_multiplier 1 \
    --batch_size 128 --workers 4 --print_freq 10 \
    --cell_file 'cells/imagenet_genotype.json' \
    --world_size $NGPUS --distributed  --dist_url 'tcp://127.0.0.1:24443' \
    --resume --resume_name 'imagenet.pth.tar'
