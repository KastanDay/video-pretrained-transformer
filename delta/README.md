

export MY_ACCOUNT=bbki-delta-cpu && \
export CONTAINER=/scratch/bbki/kastanday/apptainer_cache/whisper_latest.sif  && \
srun \
 --mem=245g \
 --nodes=1 \
 --ntasks-per-node=1 \
 --cpus-per-task=128 \
 --partition=cpu \
 --account=${MY_ACCOUNT} \
 --time=24:00:00 \
 --pty \
 singularity run --bind /scratch/bbki/kastanday:/scratch --nv ${CONTAINER} /bin/bash


# GOOD COMMAND
### no --nv for cpu partition. 
THEN deactivate all conda envs!!

export CONTAINER=/scratch/bbki/kastanday/apptainer_cache/whisper_latest.sif  && \
  singularity run --bind /scratch:/scratch ${CONTAINER} /bin/bash