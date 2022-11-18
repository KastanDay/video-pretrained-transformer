#!/bin/bash -l

#SBATCH --job-name=pdg_viz_cpu
#SBATCH --partition=cpu
#SBATCH --account=bbki-delta-cpu
#SBATCH --time=24:00:00

#SBATCH --export=ALL,RAY_worker_register_timeout_seconds=120

#SBATCH --nodes=2
#SBATCH --mem=0
#SBATCH --exclusive

set -x
## UPDATES ##
## 1. tasks-per-node used to be 1, but then I was only seeing 1 node of usage at a time.
## 2. I removed the --exclusive so it can run on multiple nodes at once.
## 3. Todo add #SBATCH --tasks=50... or something.

## Don't request full nodes, instead we're much more likely to get more availability with requesting 80-95% of a node.
## We'd rather have more nodes, with less per node anyway.
## total mem is 252g
## total cpu is 128g

# __doc_head_address_start__

# one thread per node... or per task? I think set it for total number of CPU cores... (10x60)
# In multi-worker setting, we want one thread per worker instead of many threads per worker to avoid contention.
# https://docs.ray.io/en/latest/ray-core/configure.html#cluster-resources
# OMP_NUM_THREADS=600
# export OMP_NUM_THREADS

CONTAINER=/scratch/bbki/kastanday/apptainer_cache/whisper_latest.sif


echo "This is best_cpu.slurm"
export RAY_worker_register_timeout_seconds=120

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# save nodes to file
printf "%s\n" "${nodes_array[@]}" > nodes_array_whisper.txt

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

# conda init
conda activate v3_full_viz_pipeline

# increase request limit
export RAY_max_pending_lease_requests_per_scheduling_category=2000

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray stop

# LAUNCH CONTAINER
srun --nodes=1 --ntasks=1 -w "$head_node" \
    singularity run --bind /scratch/bbki/kastanday:/scratch "$CONTAINER" /bin/bash
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head \
    --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host 0.0.0.0 --log-color true --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 1

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    srun --nodes=1 --ntasks=1 -w "$node_i" \
    ray stop
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      ulimit -n unlimited
    srun --nodes=1 --ntasks=1 -w "$node_i" \
      export RAY_worker_register_timeout_seconds=120
    
    # LAUNCH CONTAINER
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        singularity run --bind /scratch/bbki/kastanday:/scratch "$CONTAINER" /bin/bash

    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --log-color true --block &
    
    # sleep 1
    # TODO -- change num-gpus back to 0. 
    echo "WORKER $i has SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
done


sleep infinity # wait forever to presist the ray runtime

