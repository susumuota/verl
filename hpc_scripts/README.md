# Batch scripts for various HPC clusters

- [ABCI 3.0](https://docs.abci.ai/v3/)
- [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/manuals)
- [Slurm](https://slurm.schedmd.com/)

## Pre-requisites

- Follow the instructions to install verl
  - https://verl.readthedocs.io/en/latest/start/install.html#install-from-docker-image
- Find the latest verl Docker image tag from:
  - https://hub.docker.com/r/verlai/verl/tags
- Convert verl Docker image to Singularity image.

```shell
cd
mkdir -p sif
cd sif

singularity pull verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif \
    docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

cd ..
```

- Install verl
  - https://verl.readthedocs.io/en/latest/start/install.html#installation-from-docker

```shell
singularity run sif/verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif
# prompt should be changed to `Singularity> `

git clone https://github.com/susumuota/verl.git
cd verl
pip3 install --no-deps -e .
```

- Login to Hugging Face and Weights & Biases

```shell
hf auth login
wandb login
```

- Download dataset and create parquet files

```shell
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
```

- Create env.txt

```shell
cat <<EOF > ~/sif/env.txt
PYTHONUNBUFFERED=1
ROCR_VISIBLE_DEVICES=
EOF

exit
# exit from the singularity container. prompt should be back to normal
```

## Run GRPO training on ABCI 3.0

```shell
cd verl
qsub -P group hpc_scripts/grpo_qwen3_8b_abci.sh

tail -qf logs/grpo_qwen3_8b_gsm8k-{jobid}.*
```

## Run GRPO training on Slurm

```shell
cd verl
sbatch hpc_scripts/grpo_qwen3_8b_slurm.sh

tail -qf logs/grpo_qwen3_8b_gsm8k-{jobid}.*
```
