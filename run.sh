

# Grit
# python main.py --cfg configs/GRIT/peptides-func-GRIT-RRWP.yaml wandb.use False accelerator "cuda:1" optim.max_epoch 20 seed 41 dataset.dir './data'

python -m grit.task_finetune --config grit/task_finetune.yaml --output_dir results/grit/task_finetune 
