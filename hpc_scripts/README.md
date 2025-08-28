# Batch scripts for various HPC clusters

- [ABCI 3.0](https://docs.abci.ai/v3/)
- [TSUBAME 4.0](https://www.t4.cii.isct.ac.jp/manuals)
- [Slurm](https://slurm.schedmd.com/)

## Pre-requisites

- Convert verl Docker image to Singularity image

```shell
mkdir -p sif

singularity pull verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

cd ..
```

- Install verl

```shell
singularity run sif/verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif

git clone https://github.com/susumuota/verl.git
cd verl
pip3 install --no-deps -e .
```

- Login to Hugging Face and Weights & Biases

```shell
hf auth login
wandb login
```

- Download dataset and model for testing

```shell
python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k

python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2.5-0.5B-Instruct')"
```

- Create env.txt

```shell
cat <<EOF > ~/sif/env.txt
PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
EOF
