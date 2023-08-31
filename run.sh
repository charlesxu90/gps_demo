#!/bin/bash

# GPS
# python -m gps.task_finetune --config gps/task_finetune.yaml --output_dir results/gps/task_finetune2 

# Grit
# python -m grit.task_finetune --config grit/task_finetune.yaml --output_dir results/grit/task_finetune2 

# MGT
python -m mgt.task_finetune --config mgt/task_finetune.yaml --output_dir results/mgt/task_finetune
