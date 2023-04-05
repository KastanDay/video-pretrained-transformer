import jsonlines
import os
import json
import glob
import traceback
import numpy as np
from numpy import load
import lovely_tensors as lt
lt.monkey_patch()
import tqdm
import wandb
import time

# hyperparams 
MODEL_VERSION_NAME  = 'yt_pretrain_adamW_iter99000'
BATCH_NAME          = "parallel_15"
learning_rate       = 1e-4  # also good: 3e-4

BASE_DIR            = '/scratch/bbki/kastanday/whisper'
MODEL_SAVE_PATH     = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
REMOTE_WHISPER_FILE = f'{BASE_DIR}/{BATCH_NAME}_whisper_output.jsonl'
REMOTE_CLIP_DIR     = f'{BASE_DIR}/{BATCH_NAME}_clip_output'
REMOTE_SCENE_FILE   = f'{BASE_DIR}/{BATCH_NAME}_scene_output.jsonl'

# Cockpit ML Debugger
# https://cockpit.readthedocs.io/en/latest/examples/01_basic_fmnist.html
from backpack import extend
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

wandb.init(
    entity="kastan",
    project="VPT-custom-t5",
    name=MODEL_VERSION_NAME,
    config={'learning_rate'     : learning_rate,
            'batch_name'        : BATCH_NAME,
            'model_save_path'   : MODEL_SAVE_PATH,
            },
    # group=datetime_str,
    tags=['AdamW', ],
)

# Instantiate clip
import clip
import torch
MODEL_SIZE = 'ViT-L/14@336px'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.  
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
clip_instance, clip_preprocess = clip.load(MODEL_SIZE, device)

# Instantiate CustomT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5Config, AutoModelWithLMHead
if os.path.exists(MODEL_SAVE_PATH):
    # resume from checkpoint
    # todo: NEED TO GET THE LATEST CHECPOINT...... 
    print("🚨 MAJOR WARNING: Probably loading the WRONG CHECKPOINT!!! ")
    # todo: use MODEL_SAVE_PATH and find the one with the highest iteration
    t5 = T5ForConditionalGeneration.from_pretrained(f"{BASE_DIR}/MODEL_CHECKPOINTS/yt_pretrain_adamW_iter99000", torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device) # float16, True
    wandb.config.update({"starting_from_checkpoint": 0.1, "channels": 16})
else:
    t5 = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base", torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device) # float16, True
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base", return_special_tokens_mask=True)
# low_cpu_mem_usage(bool, optional) — Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. experimental.
optimizer = torch.optim.AdamW(params =  t5.parameters(), lr=learning_rate) # Typically, 1e-4 and 3e-4 work well for most problems


def log_gradient_norm():
    try: 
        start_time = time.monotonic()
        total_norm = 0
        parameters = [p for p in t5.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"total_gradient_norm": total_norm})
        # print(f"Time to log norm: {(time.monotonic() - start_time):2f} seconds") # 0.01 seconds
        return total_norm
    except Exception as e:
        print("Failed to log gradient norm: ", e)

# Initialize embeddings
one_input_shape = [1, 768, 768]
att_mask_shape = [1, 768]

input_embeds_arr = torch.zeros(one_input_shape).to(device) # .astype(np.float16)
attn_mask_arr    = torch.zeros(att_mask_shape).to(device)
attn_mask_arr[0][0] = 1
attn_mask_arr[0][1] = 1
attn_mask_arr[0][2] = 1

t5.train()
train_itr = 0
with jsonlines.open(REMOTE_SCENE_FILE, 'r') as scene_reader:
    # Zipping the scene graph with the clip + whisper embeddings
    # itr over videos
    for scene_seg_list, clip_npz_path in tqdm.tqdm(zip(scene_reader, glob.glob(os.path.join(REMOTE_CLIP_DIR, '*'), recursive = True))):
        try:
            np_loaded = np.load(clip_npz_path, allow_pickle=True)
            object_list_of_str = []
            scene_seg_list = json.loads(scene_seg_list)
        except Exception as e:
            print(f"Failed to load compressed numpy: {e}")
            continue
        
        # iterate over segments
        for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
            try:
                frame_embedding       = np_loaded[f'arr_{segment_index}'].item()['frame_embeddings']
                caption_embedding     = np_loaded[f'arr_{segment_index}'].item()['text_caption_embeddings']
                whisper_text_captions = np_loaded[f'arr_{segment_index}'].item()['captions']
                frame_embedding       = torch.from_numpy(frame_embedding.reshape((768,))).to(device)
                caption_embedding     = torch.from_numpy(caption_embedding).to(device)

                # embed whisper_text_captions with CLIP
                scene_caption = scene_seg_list[segment_index]
                scene_caption = clip.tokenize(scene_caption).to(device)
                with torch.inference_mode(): # even faster than no_grad()
                    scene_embedding = clip_instance.encode_text(scene_caption)
                scene_embedding = scene_embedding.reshape((768,))

                # Update embedding array
                input_embeds_arr[0][0] = frame_embedding
                input_embeds_arr[0][1] = caption_embedding
                input_embeds_arr[0][2] = scene_embedding
                # Set to torch
                decoder_input_embeds_arr = np.random.rand( *one_input_shape )  # .astype(np.float16) # need fp32
                
                # print("Input shapes:", scene_embedding, caption_embedding, frame_embedding)
                ''' Forward pass '''
                labels = t5_tokenizer(whisper_text_captions, return_tensors="pt").input_ids.to(device)
                outputs = t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, labels=labels)
                loss = outputs[0]
                
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
            except Exception as e:
                print("During training loop: ", e)
                print(traceback.format_exc())
                continue

print(f"✅ Finished_training_batch_{BATCH_NAME}")
t5.save_pretrained(f"{BASE_DIR}/MODEL_CHECKPOINTS/Finished_training_batch_{BATCH_NAME}")
