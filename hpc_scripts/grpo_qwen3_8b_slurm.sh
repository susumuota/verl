#!/bin/bash
#SBATCH --job-name=grpo_qwen3_8b_gsm8k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1200G
#SBATCH --partition=your-partition
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=240
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# this file is copied and modified from examples/slurm/ray_on_slurm.slurm
# run the following command to see the difference:
#   diff -u examples/slurm/ray_on_slurm.slurm hpc_scripts/ppo_slurm.sh

# singularity image file can be converted from the docker image
# see https://hub.docker.com/r/verlai/verl/tags
# run the following command to convert the image:
#   singularity pull verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif docker://verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

# load necessary modules
module purge
module load cuda/12.8

# to avoid MemoryError
ulimit -l unlimited
ulimit -m unlimited
ulimit -v unlimited

# replace these information with your own
verl_workdir=${HOME}/verl
train_files=${HOME}/data/gsm8k/train.parquet
val_files=${HOME}/data/gsm8k/test.parquet
sif_file=${HOME}/sif/verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif
env_file=${HOME}/sif/env.txt
model_name=Qwen/Qwen3-8B
project_name=verl_grpo
experiment_name=qwen3_8b_gsm8k
# replace these information with your own

# define HPC_* instead of SLURM_*
HPC_NNODES=$SLURM_NNODES
HPC_GPUS_ON_NODE=$SLURM_GPUS_PER_NODE
HPC_CPUS_ON_NODE=$SLURM_CPUS_PER_TASK

# Getting the node names
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
worker_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)

head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization

printenv

echo "Starting HEAD at $head_node"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --env-file "$env_file" --nv --bind "$verl_workdir" "$sif_file" \
        ray start --head --node-ip-address="$head_node_ip" --port="$port" \
            --num-cpus "${HPC_CPUS_ON_NODE}" --num-gpus "${HPC_GPUS_ON_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 20

for worker_node in $worker_nodes; do
    echo "Starting WORKER at $worker_node"
    srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
        singularity exec --env-file "$env_file" --nv --bind "$verl_workdir" "$sif_file" \
            ray start --address "$ip_head" --num-cpus "${HPC_CPUS_ON_NODE}" --num-gpus "${HPC_GPUS_ON_NODE}" --block &
    sleep 10
done

sleep 10

echo "Confirming status at $head_node"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --env-file "$env_file" --nv --bind "$verl_workdir" "$sif_file" \
        ray status --address "$ip_head"

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --env-file "$env_file" --nv --bind "$verl_workdir" "$sif_file" \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_name" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$project_name" \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="${HPC_GPUS_ON_NODE}" \
    trainer.nnodes="${HPC_NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15
