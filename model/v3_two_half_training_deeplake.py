import os
import traceback

import deeplake as dl
import lovely_tensors as lt
import numpy as np
import torch
import tqdm
import wandb
from termcolor import colored
from tqdm import tqdm

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalCall=false
# ^^ due to not understanding ray

# pip install transformers "deeplake[enterprise]" wandb lovely-tensors  pandas termcolor sentencepiece
# not 100% necessary ofr this file: "ray[default]==2.2.0"

# print(colored(f"👉 Using only CPU!!", "cyan", attrs=["reverse", "bold"]))
# device = 'cpu'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

lt.monkey_patch()
# hyperparams
MODEL_VERSION_NAME = 'yt_pretrain_half_half'
learning_rate = 1e-4  # also good: 3e-4

BATCH_NAME = "parallel_15"
# BASE_DIR = '/scratch/bbki/kastanday/whisper'
# BASE_DIR = '/mnt/storage_ssd'
BASE_DIR = '~/VPT/'
MODEL_SAVE_PATH = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
DATABASE_FILEPATH = f'{BASE_DIR}/v4_CLIP_encode_results_{BATCH_NAME}'

# Instantiate CustomT5
from transformers import (AutoModelWithLMHead, T5Config,
                          T5ForConditionalGeneration, T5Model, T5Tokenizer)

if os.path.exists(MODEL_SAVE_PATH):
    # resume from checkpoint
    # todo: NEED TO GET THE LATEST CHECPOINT......
    print("🚨 MAJOR WARNING: Probably loading the WRONG CHECKPOINT!!! ")
    # todo: use MODEL_SAVE_PATH and find the one with the highest iteration
    t5 = T5ForConditionalGeneration.from_pretrained(
        f"{BASE_DIR}/MODEL_CHECKPOINTS/yt_pretrain_adamW_iter99000",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False).to(device)  # float16, True
    # wandb.config.update({"starting_from_checkpoint": 0.1, "channels": 16})
else:
    t5 = T5ForConditionalGeneration.from_pretrained(
        "google/t5-v1_1-large",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False).to(device)  # float16, True
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-large",
                                           return_special_tokens_mask=True)
# low_cpu_mem_usage(bool, optional) — Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. experimental.
optimizer = torch.optim.AdamW(
    params=t5.parameters(),
    lr=learning_rate)  # Typically, 1e-4 and 3e-4 work well for most problems
print("Done loading model and optimizer")


def log_gradient_norm():
    try:
        # start_time = time.monotonic()
        total_norm = 0
        parameters = [
            p for p in t5.parameters()
            if p.grad is not None and p.requires_grad
        ]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm**0.5
        wandb.log({"total_gradient_norm": total_norm})
        # print(f"Time to log norm: {(time.monotonic() - start_time):2f} seconds") # 0.01 seconds
        return total_norm
    except Exception as e:
        print("Failed to log gradient norm: ", e)


# Initialize embeddings
one_input_shape = [1, 1024, 1024]
att_mask_shape = [1, 1024]

input_embeds_arr = torch.zeros(one_input_shape).to(
    device)  # .astype(np.float16)
attn_mask_arr = torch.zeros(att_mask_shape).to(device)

t5.train()
train_itr = 0

wandb.init(
    entity="kastan",
    project="VPT-custom-t5",
    name=MODEL_VERSION_NAME,
    config={
        'learning_rate': learning_rate,
        'batch_name': BATCH_NAME,
        'model_save_path': MODEL_SAVE_PATH,
    },
    # group=datetime_str,
    tags=[
        'AdamW',
    ],
)

print("About to load dataset")
ds = dl.load(DATABASE_FILEPATH)
# iterate over segments
print("about to iterate over segments")
for i, segment in tqdm(
        enumerate(ds),
        desc='training',
        total=ds.max_len,
        bar_format=
        '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
):
    try:
        # if has clip embedding
        if segment.clip_pooled_embedding.numpy().any():
            clip_pooled_embedding = torch.from_numpy(
                segment.clip_pooled_embedding.numpy()).to(device)
            clip_last_hidden_states = torch.from_numpy(
                segment.clip_last_hidden_states.numpy()).to(device)
            caption_embedding = torch.from_numpy(
                segment.caption_embedding.numpy()).to(device)
            whisper_text_captions = segment.caption.data()['value']

            # split caption embedding into two halves
            s = caption_embedding.shape[0]
            s_half = s // 2
            caption_embedding_first_half = caption_embedding[:s_half]
            full_caption_tokenized = t5_tokenizer(
                whisper_text_captions,
                return_tensors="pt").input_ids.to(device)
            # only keep 2nd half of caption to use as labels.
            labels = full_caption_tokenized[0][:s_half].reshape(1, -1)

            # Update input embedding array
            input_embeds_arr[0][0] = clip_pooled_embedding  # shape (577, 1024)
            input_embeds_arr[0][
                1:578] = clip_last_hidden_states  # shape (577, 1024)
            input_embeds_arr[0][579:579 +
                                s_half] = caption_embedding_first_half

            attn_mask_arr[0][0:579 + s_half] = 1
            # print("Total inputs", 579 + s_half)
            assert 579 + s_half <= 1024, print("Too many inputs")

            # print("Input shapes:",)
            # print('clip_pooled_embedding', clip_pooled_embedding)
            # print('clip_last_hidden_states', clip_last_hidden_states)
            # print('caption_embedding_first_half', caption_embedding_first_half)
            # print('input_embeds_arr', input_embeds_arr)
            ''' Forward pass '''
            outputs = t5.forward(inputs_embeds=input_embeds_arr,
                                 attention_mask=attn_mask_arr,
                                 labels=labels)
            loss = outputs[0]
            print("😁 GOT LOSS: ", loss)
            ''' Backwards pass '''
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            train_itr += 1
            # logging
            if train_itr % 50 == 0:
                wandb.log({"loss": loss})
                log_gradient_norm()
            # save checkpoints
            if train_itr % 1500 == 0:
                save_path = f"{MODEL_SAVE_PATH}_iter{train_itr}"
                t5.save_pretrained(save_path)
                print("SAVED MODEL CHECKPOINT TO: ", save_path)
        else:
            print("No Clip at index", i)
    except Exception as e:
        print("During training loop: ", e)
        print(traceback.format_exc())
        continue

print(f"✅ Finished_training_batch_{BATCH_NAME}")
t5.save_pretrained(
    f"{BASE_DIR}/MODEL_CHECKPOINTS/Finished_training_batch_{BATCH_NAME}")
