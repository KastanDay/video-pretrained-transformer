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
from modeling_vpt_in_mosaicml import VPT_model  # original work
from modeling_vpt_in_mosaicml import vpt_transform_dataset_to_batch
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

  BASE_DIR = ''
  if "gpu" in hostname or "gpu" in hostname:
    print("Hostname: Delta GPU")
    BASE_DIR = '/scratch/bbki/kastanday/whisper'
  elif 'storage' in hostname:
    print("Hostname: kastan storage")
    BASE_DIR = '/mnt/storage_ssd'
  elif 'dgx' in hostname:
    print("Hostname: DGX")
    BASE_DIR = '/raid/projects/kastan'
  elif 'hal' in hostname:
    print("Hostname: HAL")
    BASE_DIR = '~/thesis/VPT_data'
  elif '103ai' in hostname:
    print("Hostname: 103ai, EPYC")
    # BASE_DIR = '/mnt/teton/vpt/data/deeplake_parallel_15'
    BASE_DIR = '/mnt/teton/vpt/data/deeplake_handpicked'

  ##### MANUAL PARAMS #####
  '''
  hyperparam experiments:
  1. model name: google/t5-large, google/t5-v1_1-large, google/flan-t5-large
  2. learning rate
  3. cosine annealing lr scheduler (warmup... and max lr)?
  '''

  MODEL_VERSION_NAME = 'feb_24_half_half_handpicked_sweep_0'

  sweep_configuration = {
      'method': 'random',
      'name': 'sweep',
      'metric': {
          'goal': 'minimize',
          'name': 'val_loss_cross_entropy'
      },
      'parameters': {
          'learning_rate': {
              'values': [5e-5, 1e-4, 3e-4, 1e-3]
          },
          'cosine_warmup_batches': {
              'values': [500, 2000, 8000]
          },
          'model_huggingface_name': {
              'values': ["google/t5-large", "google/t5-v1_1-large", "google/flan-t5-large"]
          },
          'batch_size': 1,
          'base_dir': BASE_DIR,
          'model_version_name': MODEL_VERSION_NAME,
          'batch_name': "handpicked_downloads",
          'dataset_path': f'{BASE_DIR}/CLIP_encode_results_{BATCH_NAME}',
          'model_save_path': f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}',
          'max_run_to_try': 30,
      }
  }

  #### START SWEEEP ####

  sweep_id = wandb.sweep(sweep=sweep_configuration, project="project-name")
  wandb.agent(sweep_id=sweep_id, count=sweep_configuration['parameters']['max_run_to_try'], function=train)


def train():

  with wandb.init(config=config) as run:
    config = wandb.config
    # create dataloader
    ds = dl.load(DATABASE_FILEPATH)
    columns_for_training = ['clip_pooled_embedding', 'caption_embedding', 'clip_last_hidden_states', 'caption']

    train_dataloader = ds.pytorch(
        tensors=columns_for_training,
        transform=vpt_transform_dataset_to_batch,
        num_workers=psutil.cpu_count(),
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        use_local_cache=False,  # downloads to ~/.deeplake, good when using S3.
    )

    model = VPT_model(model_huggingface_name=config.model_huggingface_name, model_version_name=config.model_version_name)

    # todo: Maybe Adafactor is the better optimizer? See discussion https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
    # Training without LR warmup or clip_threshold is not recommended.
    # use scheduled LR warm-up to fixed LR
    # FOR EX:
    # from transformers.optimization import Adafactor, AdafactorSchedule
    # optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # lr_scheduler = AdafactorSchedule(optimizer)
    # optimizer = transformers.Adafactor(params=model.parameters(), lr=0.001, scale_parameter=False, relative_step=False)
    # fsdp_config['min_params']

    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.learning_rate)  # Typically, 1e-4 and 3e-4 work well for most problems
    wandb_logger = WandBLogger(
        init_kwargs={
            "config": {
                'learning_rate': config.learning_rate,
                'batch_name': config.batch_name,
                'dataset_path': config.database_filepath,
                'model_save_path': config.model_save_path,
            },
            "entity": "kastan",
            "project": "VPT-custom-t5",
            "name": config.model_version_name,
            # group=datetime_str,
            "tags": [
                # 'Adafactor w/ auto_lr',
                'Adamw',
                # 'FSDP',
                'MosaicML',
                'resume_from_checkpoint',
            ],
        })

    ## FSDP config: https://docs.mosaicml.com/en/v0.11.1/notes/distributed_training.html#composer-s-fsdp-auto-wrap-policy
    # If any module has more parameters than fsdp_config['min_params'], it will be wrapped.

    # fsdp_config = {
    #     'sharding_strategy': 'FULL_SHARD',
    #     'min_params': 1e8,
    #     'mixed_precision': 'DEFAULT',
    #     'backward_prefetch': 'BACKWARD_POST',
    #     'activation_checkpointing': False,
    #     'verbose': True
    # }

    cosine_lr_schedule = optim.CosineAnnealingWithWarmupScheduler(t_warmup=Time.from_batch(config.cosine_warmup_batches))

    # alibi attention https://docs.mosaicml.com/en/v0.11.1/method_cards/alibi.html
    # only directly supported for GPT and BERT.
    # alibi = Alibi(
    #     max_sequence_length=1024,
    #     train_sequence_length_scaling=0.25,
    # )

    # Instantiate the trainer
    composer_trace_dir = 'composer_profiler'
    torch_trace_dir = 'torch_profiler'

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=train_dataloader,
        optimizers=optimizer,
        max_duration=3,  # epochs
        device="gpu" if torch.cuda.is_available() else "cpu",
        run_name=f'{config.model_version_name}',
        save_folder=f"{config.base_dir}/{config.model_version_name}/checkpoints",
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
        # fsdp_config=fsdp_config,

        # ----------- RESUME FROM CHECKPOINT -----------
        save_overwrite=True,
        # load_path=f"{MODEL_SAVE_PATH}/latest-rank0.pt",  # resume from checkpoint
        # "/raid/projects/kastan/mosaic_yt_pretrain_half_half/checkpoints/latest-rank0.pt",  # resume from checkpoint

        # ----------- PROFILE -----------
        # train_subset_num_batches=16,
        # profiler=Profiler(
        #     trace_handlers=[JSONTraceHandler(folder=composer_trace_dir, overwrite=True)],
        #     schedule=cyclic_schedule(
        #         wait=0,
        #         warmup=1,
        #         active=4,
        #         repeat=1,
        #     ),
        #     torch_prof_folder=torch_trace_dir,
        #     torch_prof_overwrite=True,
        # ),
        seed=42)
    trainer.fit()


if __name__ == "__main__":
  main()
