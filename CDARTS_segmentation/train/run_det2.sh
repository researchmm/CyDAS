export DETECTRON2_DATASETS="/home/hongyuan/data/"
NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_autos4_det2.py --world_size $NGPUS --seed 12367