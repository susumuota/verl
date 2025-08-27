#!/bin/bash
#SBATCH --job-name=verl-ray-on-slurm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1200G
#SBATCH --partition=your-partition
#SBATCH --time=01:00:00
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
apptainer_command=singularity
apptainer_image_path=${HOME}/sif/verl-app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2.sif
apptainer_env_file=${HOME}/sif/env.txt
# replace these information with your own

# Getting the node names
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
worker_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2)

head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization

printenv

echo "Starting HEAD at $head_node"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    $apptainer_command exec --env-file $apptainer_env_file --nv --bind $verl_workdir $apptainer_image_path \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

for worker_node in $worker_nodes; do
    echo "Starting WORKER at $worker_node"
    srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
        $apptainer_command exec --env-file $apptainer_env_file --nv --bind $verl_workdir $apptainer_image_path \
            ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
    sleep 5
done

echo "Confirming status at $head_node"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    $apptainer_command exec --env-file $apptainer_env_file --nv --bind $verl_workdir $apptainer_image_path \
        ray status --address "$ip_head"

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    $apptainer_command exec --env-file $apptainer_env_file --nv --bind $verl_workdir $apptainer_image_path \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.name=vllm \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.use_kl_in_reward=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=verl_ppo \
    trainer.experiment_name=Qwen2.5-0.5B-Instruct-PPO-GSM8K \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${SLURM_GPUS_ON_NODE}" \
    trainer.nnodes="${SLURM_NNODES}" \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 2>&1 | tee verl_demo_slurm.log
