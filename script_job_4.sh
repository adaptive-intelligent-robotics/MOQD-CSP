#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=msw16 # required to send email notifcations - please replace <your_username> with your college login n>
export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH
source activate
source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh

#python3 experiment_from_config_cma.py configs/0823/cma_50_relaxation_lr1_sigma_1_with_threshold_batch_10.json
python3 experiment_from_config.py configs/0826/other_material_4.json
#python3 debugger.py

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
