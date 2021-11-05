#!/usr/bin/env bash


GPUs=8
CONFIG='configs/CyDAS_retinanet_1x.py'
WORKDIR='./work_dirs/CyDAS_1x/'

python -m torch.distributed.launch \
--nproc_per_node=${GPUs} train.py \
--validate \
--gpus ${GPUs} \
--launcher pytorch \
--config ${CONFIG} \
--work_dir ${WORKDIR}
