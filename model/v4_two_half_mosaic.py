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
import subprocess

import deeplake as dl
import lovely_tensors as lt
import numpy as np
import torch
import transformers
import wandb
from composer import Trainer
from composer.loggers import WandBLogger
from composer.models import HuggingFaceModel
from modeling_vpt_in_mosaicml import VPT_model  # original work
from termcolor import colored
from tqdm import tqdm
import transformers

lt.monkey_patch()

# device = "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)
model_huggingface_name = "google/t5-v1_1-large"

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

# hyperparams
MODEL_VERSION_NAME = 'mosaic_yt_pretrain_half_half'
learning_rate = 1e-4  # recc for t5: 1e-4 and 3e-4
BATCH_NAME = "parallel_15"
MODEL_SAVE_PATH = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
# DATABASE_FILEPATH = f'{BASE_DIR}/v4_CLIP_encode_results_{BATCH_NAME}'
DATABASE_FILEPATH = f'{BASE_DIR}/shorter_v4_CLIP_encode_results_{BATCH_NAME}'


def main():
  # create dataloader
  ds = dl.load(DATABASE_FILEPATH)
  columns_for_training = ['clip_pooled_embedding', 'caption_embedding', 'clip_last_hidden_states', 'caption']
  train_dataloader = ds.pytorch(tensors=columns_for_training,
                                transform=my_dataloader_batching_transform,
                                num_workers=0,
                                batch_size=1,
                                pin_memory=False,
                                shuffle=False,
                                drop_last=False)

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

  optimizer = torch.optim.AdamW(params=model.parameters(),
                                lr=learning_rate)  # Typically, 1e-4 and 3e-4 work well for most problems
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

  # todo: implement evaluation or something on a holdout/validation set. Maybe yt1b val.
  # todo: implement model saving
  # todo: implement LR scheduler.... cosine decay with warmup.

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
      save_interval="500ba",  # 2k batches
      save_num_checkpoints_to_keep=5,
      # grad_accum=1,
      # eval_interval=100,
      loggers=[wandb_logger],
      precision='fp32',
      # fsdp_config=fsdp_config,
      # overwrite=True,  # existing checkpoints overwritten
      # save_folder="s3://my-bucket/{run_name}/checkpoints",
      # save_filename="ep{epoch}.pt",
      # save_overwrite=True,
      load_path=
      "/raid/projects/kastan/mosaic_yt_pretrain_half_half/checkpoints/latest-rank0.pt",  # resume from checkpoint
      seed=42)
  trainer.fit()


from transformers import T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained(model_huggingface_name, return_special_tokens_mask=True)


def my_dataloader_batching_transform(segment_batch):
  '''
  param: segment_batch: IterableOrderedDict. 1 SEGMENT (not batch_size, just one).
  We put a bunch of these together to get a btach. 
  
  returns: batch dictionary.  Keys: input_embeds_arr, attn_mask_arr, labels_tokenized
                              Values: batched Torch Tensors of shape <1, 1024, 1024>. These are stacked to create a batch.
  '''
  # print("👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉 SEGMENT BATCH", flush=True)
  batch = {}  # keys: input_embeds_arr, attn_mask_arr, labels

  # Loop over BATCH_SIZE. Create dictionary where key = name, value = batched tensor
  for key, numpy_array in zip(segment_batch.keys(), segment_batch):
    # print("------------- PRINTING SEGMENT --------- ")
    if key == 'clip_pooled_embedding':
      # print("⭐️1️⃣ pooled embedding")
      numpy_array = numpy_array.reshape(1, -1)  # for batch size purposes.
      if key in batch.keys():
        batch[key] = torch.cat((batch[key], torch.from_numpy(numpy_array).to(device)), dim=0)
      else:
        batch[key] = torch.from_numpy(numpy_array).to(device)

    elif key == 'caption_embedding':
      # print("⭐️2️⃣ caption embedding")
      # keep only the first HALF of caption embedding.
      caption_length = numpy_array.shape[0]
      s_half = caption_length // 2
      # constant length of 446, pad with zeros. 446 is the max length of a caption (1024 - 577 - 1).
      caption_embedding_full_length = torch.zeros((446, 1024)).to(device)
      caption_embedding_full_length[0:s_half] = torch.from_numpy(numpy_array[0:s_half]).to(device)

      # setup attention mask now that we know full length of caption
      att_mask_shape = [1024]
      attn_mask_arr = torch.zeros(att_mask_shape).to(device)
      attn_mask_arr[0:578 + s_half] = 1

      if key in batch.keys():
        # batch[key] = torch.cat((batch[key], torch.ones(10)), dim=0) # todo BAD for testing
        batch[key] = torch.cat((batch[key], caption_embedding_full_length), dim=0)
        batch['attn_mask_arr'] = torch.cat((batch['attn_mask_arr'], attn_mask_arr), dim=0)
      else:
        # batch[key] = torch.ones(10)  # todo BAD for testing
        batch[key] = caption_embedding_full_length
        batch['attn_mask_arr'] = attn_mask_arr

    elif key == 'clip_last_hidden_states':
      # print("⭐️3️⃣ clip last hidden states")
      if key in batch.keys():
        batch[key] = torch.cat((batch[key], torch.from_numpy(numpy_array).to(device)), dim=0)
      else:
        batch[key] = torch.from_numpy(numpy_array).to(device)

    elif key == 'caption':
      caption = numpy_array[0]  # passed in as a single-element list.
      # print("⭐️4️⃣ CAPTION")
      # print(caption)
      full_caption_tokenized = t5_tokenizer(caption, padding=False, truncation=True,
                                            return_tensors="pt").input_ids.to(device)
      caption_length = full_caption_tokenized.shape[1]
      s_half = caption_length // 2
      # only keep 2nd half of caption to use as labels.
      proper_shape = full_caption_tokenized[0][s_half:].shape[0]
      labels_full_length = torch.ones((512), dtype=torch.int64).to(device) * -100
      labels_full_length[:proper_shape] = full_caption_tokenized[0][s_half:]  # 👈 take 2nd half!!
      if 'labels' in batch.keys():
        batch['labels'] = torch.cat((batch['labels'], labels_full_length), dim=0)
      else:
        batch['labels'] = labels_full_length
  return batch


if __name__ == "__main__":
  main()