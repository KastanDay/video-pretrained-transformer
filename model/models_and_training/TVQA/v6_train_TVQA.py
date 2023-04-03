'''
Requirements:
pip install mosaicml transformers "deeplake[enterprise]" wandb lovely-tensors  pandas termcolor sentencepiece
not 100% necessary for this file: "ray[default]==2.2.0"


pyright: reportGeneralTypeIssues=false
^^ due to not understanding deeplake
pyright: reportPrivateImportUsage=false
pyright: reportOptionalMemberAccess=false
pyright: reportOptionalCall=false
^^ due to not understanding ray
'''
import os
import subprocess
import traceback

import composer
import deeplake as dl
import lovely_tensors as lt
import numpy as np
import psutil
import torch
import transformers
import wandb
from composer import Time, Trainer, optim
from composer.algorithms import Alibi, FusedLayerNorm
from composer.loggers import WandBLogger
from composer.models import HuggingFaceModel
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from modeling_vpt_for_TVQA import VPT_model  # original work
from termcolor import colored

lt.monkey_patch()

global BASE_DIR
global BATCH_NAME
global MODEL_VERSION_NAME
global MODEL_SAVE_PATH
global DATABASE_FILEPATH


def main():
  print("CPU count:", psutil.cpu_count())
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  print("Running on:", device)

  result = subprocess.run(["hostname"], capture_output=True, text=True)
  hostname = str(result.stdout.strip())

  global BASE_DIR
  BASE_DIR = ''
  if '103ai' in hostname:
    print("Hostname: 103ai, EPYC")
    # BASE_DIR = '/mnt/teton/vpt/data/deeplake_parallel_15'
    BASE_DIR = '/mnt/teton/vpt/'
  elif "gpu" in hostname or "gpu" in hostname:
    print("Hostname: Delta GPU")
    BASE_DIR = '/scratch/bbki/kastanday/whisper'
  elif 'dgx' in hostname:
    print("Hostname: DGX")
    BASE_DIR = '/raid/projects/kastan'
  elif 'hal' in hostname:
    print("Hostname: HAL")
    BASE_DIR = '~/thesis/VPT_data'

  train()


def train():

  #PARAMS
  batch_name = 'tvqa_whole'
  DATABASE_FILEPATH = f'/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/mar_28_TVQA_encode_{batch_name}'
  model_save_path = f'{BASE_DIR}/data/benchmark_datasets/TVQA/CHECKPOINTS'

  model_version_name = 'first_attempt'
  model_huggingface_name = "google/flan-t5-large"

  batch_size = 1
  learning_rate = 1e-3
  cosine_warmup_batches = 10_000  # from sweep

  model = VPT_model(model_huggingface_name=model_huggingface_name,)

  # create dataloader
  ds = dl.load(DATABASE_FILEPATH)
  columns_for_training = ['context_vector', 'label']
  train_dataloader = ds.pytorch(
      tensors=columns_for_training,
      transform=model.vpt_transform_dataset_to_batch,
      num_workers=psutil.cpu_count(),
      batch_size=batch_size,
      pin_memory=True,
      shuffle=False,
      drop_last=False,
      use_local_cache=False,  # downloads to ~/.deeplake, good when using S3.
  )

  optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)  # Typically, 1e-4 and 3e-4 work well for most problems
  wandb_logger = WandBLogger(
      init_kwargs={
          "config": {
              'learning_rate': learning_rate,
              'batch_name': batch_name,
              'dataset_path': DATABASE_FILEPATH,
              'model_save_path': model_save_path,
          },
          "entity": "kastan",
          "project": "VPT-custom-t5",
          "name": model_version_name,
          # group=datetime_str,
          "tags": [
              # 'Adafactor w/ auto_lr',
              'Adamw',
              # 'FSDP',
              'MosaicML',
              'resume_from_checkpoint',
          ],
      })

  cosine_lr_schedule = optim.CosineAnnealingWithWarmupScheduler(t_warmup=Time.from_batch(cosine_warmup_batches))

  trainer = Trainer(
      model=model,
      train_dataloader=train_dataloader,
      eval_dataloader=train_dataloader,
      optimizers=optimizer,
      max_duration=3,  # epochs
      device="gpu" if torch.cuda.is_available() else "cpu",
      run_name=f'{model_version_name}',
      save_folder=f"{BASE_DIR}/{model_version_name}/checkpoints",
      save_interval="1000ba",  # 2k batches
      # save_filename="ep{epoch}.pt",
      save_num_checkpoints_to_keep=2,
      schedulers=[cosine_lr_schedule],
      loggers=[wandb_logger],
      device_train_microbatch_size='auto',
      precision='amp_bf16',  # also works: fp32
      # eval_interval=0,

      # grad_accum=10, # requires multiple GPUs I guess
      # algorithms=[alibi],  # FusedLayerNorm() -- use NGC
      # algorithms=[FusedLayerNorm()],

      # ----------- RESUME FROM CHECKPOINT -----------
      save_overwrite=True,
      # load_path=f"{MODEL_SAVE_PATH}/latest-rank0.pt",  # resume from checkpoint
      seed=42)
  trainer.fit()


if __name__ == "__main__":
  main()
