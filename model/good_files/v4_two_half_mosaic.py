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

print("CPU count:", psutil.cpu_count())
# device = "cpu"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

result = subprocess.run(["hostname"], capture_output=True, text=True)
hostname = str(result.stdout.strip())

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
    BASE_DIR = '~/thesis/VPT_data/'
elif '103ai' in hostname:
    print("Hostname: 103ai, EPYC")
    # BASE_DIR = '/mnt/teton/vpt/data/'
    # BASE_DIR = '/mnt/teton/vpt/data/deeplake_parallel_15/'
    BASE_DIR = '/mnt/teton/vpt/data/deeplake_handpicked/'

# MANUAL PARAMS
learning_rate = 1e-4  # recc for t5: 1e-4 and 3e-4
batch_size = 1
model_huggingface_name = "google/t5-v1_1-large"

MODEL_VERSION_NAME = 'feb_17_mosaic_yt_pretrain_half_half_handpicked'
BATCH_NAME = "handpicked_downloads"
MODEL_SAVE_PATH = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
# DATABASE_FILEPATH = f'{BASE_DIR}/v4_CLIP_encode_results_{BATCH_NAME}'
DATABASE_FILEPATH = f'{BASE_DIR}/CLIP_encode_results_{BATCH_NAME}'


def main():
    # create dataloader
    ds = dl.load(DATABASE_FILEPATH)
    columns_for_training = ['clip_pooled_embedding', 'caption_embedding', 'clip_last_hidden_states', 'caption']

    # todo: remove all mention of cuda from this functino (no .todevice), then use pin=True.
    train_dataloader = ds.pytorch(
        tensors=columns_for_training,
        transform=vpt_transform_dataset_to_batch,
        num_workers=psutil.cpu_count(),  # why can't I change this... 
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        use_local_cache=False,  # downloads to ~/.deeplake
    )

    # todo: use c++ dataloader "deeplake[enterprise]"
    # train_loader = ds.dataloader()\
    #              .transform(transform)\
    #              .batch(32)\
    #              .shuffle(True)\
    #              .pytorch(tensors=['images', 'labels'], num_workers = 8)

    # run training with our model
    # todo: implement evaluation or something on a holdout/validation set. Maybe yt1b val.
    model = VPT_model(model_huggingface_name=model_huggingface_name, model_version_name=MODEL_VERSION_NAME)

    # adafactor setup as suggested here: https://discuss.huggingface.co/t/t5-finetuning-tips/684/3
    # todo: Critically implement LR warmup if using adafactor.
    # Training without LR warmup or clip_threshold is not recommended.
    # use scheduled LR warm-up to fixed LR
    # FOR EX:
    # from transformers.optimization import Adafactor, AdafactorSchedule
    # optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # lr_scheduler = AdafactorSchedule(optimizer)
    # optimizer = transformers.Adafactor(params=model.parameters(), lr=0.001, scale_parameter=False, relative_step=False)
    # fsdp_config['min_params']

    # all params are fp32
    # for param in model.parameters():
    #   print(param.dtype)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)  # Typically, 1e-4 and 3e-4 work well for most problems
    wandb_logger = WandBLogger(
        init_kwargs={
            "config": {
                'learning_rate': learning_rate,
                # 'batch_name': BATCH_NAME,
                'model_save_path': MODEL_SAVE_PATH,
            },
            "entity": "kastan",
            "project": "VPT-custom-t5",
            "name": MODEL_VERSION_NAME,
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

    fsdp_config = {
        'sharding_strategy': 'FULL_SHARD',
        'min_params': 1e8,
        'mixed_precision': 'DEFAULT',
        'backward_prefetch': 'BACKWARD_POST',
        'activation_checkpointing': False,
        'verbose': True
    }

    cosine_lr_schedule = optim.CosineAnnealingWithWarmupScheduler(t_warmup=Time.from_batch(2000))

    # alibi attention https://docs.mosaicml.com/en/v0.11.1/method_cards/alibi.html
    # only directly supported for GPT and BERT.
    # alibi = Alibi(
    #     max_sequence_length=1024,
    #     train_sequence_length_scaling=0.25,
    # )

    # Instantiate the trainer
    composer_trace_dir = 'composer_profiler'
    torch_trace_dir = 'torch_profiler'

    # trainer_device = 'cpu'
    trainer_device = 'gpu'
    print(f"Running Trainer on {trainer_device}")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=train_dataloader,
        optimizers=optimizer,
        max_duration=3,  # epochs
        device=trainer_device,  # "gpu" if torch.cuda.is_available() else "cpu",
        run_name=f'{MODEL_VERSION_NAME}',
        save_folder=f"{BASE_DIR}/{MODEL_VERSION_NAME}/checkpoints",
        save_interval="1000ba",  # 2k batches
        save_num_checkpoints_to_keep=2,
        schedulers=[cosine_lr_schedule],
        # algorithms=[alibi],  # FusedLayerNorm() -- use NGC
        loggers=[wandb_logger],
        # grad_accum=10, # requires multiple GPUs I guess
        device_train_microbatch_size='auto',
        # eval_interval=0,
        precision='amp_bf16',  # working: fp32
        # fsdp_config=fsdp_config,
        # save_folder="s3://my-bucket/{run_name}/checkpoints",
        # save_filename="ep{epoch}.pt",
        save_overwrite=True,
        # load_path=f"{MODEL_SAVE_PATH}/latest-rank0.pt",  # resume from checkpoint
        # "/raid/projects/kastan/mosaic_yt_pretrain_half_half/checkpoints/latest-rank0.pt",  # resume from checkpoint
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
