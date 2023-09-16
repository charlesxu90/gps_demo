#!/bin/bash

# GPS
# python -m gps.task_finetune --config gps/task_finetune.yaml --output_dir results/gps/task_finetune2 

# Grit
# python -m grit.task_finetune --config grit/task_finetune.yaml --output_dir results/grit/task_finetune2 

# MGT
# python -m mgt.task_finetune --config mgt/task_finetune.yaml --output_dir results/mgt/task_finetune

# Mole-BERT
# python -m mole-bert.train_vae --config mole-bert/train_vae.yaml --output_dir results/mole-bert/train_vae2
# python -m mole-bert.pretrain --config mole-bert/pretrain.yaml --output_dir results/mole-bert/pretrain --vae_ckpt results/mole-bert/train_vae/model_38_0.121.pt
# python -m mole-bert.task_finetune --config mole-bert/task_finetune.yaml --output_dir results/mole-bert/task_finetune  --ckpt_cl results/mole-bert/pretrain/model_1_-3.287.pt 
# python -m mole-bert.task_finetune --config mole-bert/task_finetune.yaml --output_dir results/mole-bert/task_finetune  --ckpt_gnn results/mole-bert/Mole-BERT.pth 

# GINE
# python -m gine.pretrain --config gine/pretrain.yaml --output_dir results/gine/pretrain
# python -m gine.pretrain_graphpred --config gine/pretrain_graphpred.yaml --output_dir results/gine/pretrain_graphpred --ckpt_pretrain results/gine/pretrain/pretrain_contextpred2.pth
# python -m gine.task_finetune --config gine/task_finetune.yaml --output_dir results/gine/task_finetune --ckpt_pretrain  results/gine/pretrain_graphpred/model_final_0.692.pt

# GCN
python -m gcn.task_finetune --config gcn/task_finetune.yaml --output_dir results/gcn/task_finetune
