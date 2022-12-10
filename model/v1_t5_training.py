import jsonlines
import os
import json
import glob
import numpy as np
import pathlib
from numpy import load
import lovely_tensors as lt
lt.monkey_patch()
import tqdm
import wandb
wandb.init("custom-t5")

# Cockpit ML Debugger
# https://cockpit.readthedocs.io/en/latest/examples/01_basic_fmnist.html
from backpack import extend
from cockpit import Cockpit, CockpitPlotter
from cockpit.utils.configuration import configuration

dir_name = "parallel_15"
REMOTE_WHISPER_FILE = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{dir_name}_whisper_output.jsonl'
REMOTE_CLIP_DIR  = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{dir_name}_clip_output'
REMOTE_SCENE_FILE = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{dir_name}_scene_output.jsonl'

# Instantiate clip
import clip
import torch
MODEL_SIZE = 'ViT-L/14@336px'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.  
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_instance, clip_preprocess = clip.load(MODEL_SIZE, device)


'''
T5 MODEL SELECTION

T5 V1.1 --  https://huggingface.co/docs/transformers/model_doc/t5v1.1 && https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511
small - base - large - 3b/xl - 11b/xxl

OG: t5-small

'google/t5-base-lm-adapt' # largest on my server (without float16)
'google/t5-xl-lm-adapt'

google/t5-v1_1-large
'''

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5Config, AutoModelWithLMHead
# MODEL_SIZE = "t5-base"
MODEL_NAME = "google/t5-v1_1-base"
# MODEL_NAME = "google/t5-base-lm-adapt"
# config = T5Config.from_pretrained(MODEL_NAME)
t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device) # float16, True
t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, return_special_tokens_mask=True)
# low_cpu_mem_usage(bool, optional) — Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. experimental.
optimizer = torch.optim.Adam(params =  t5.parameters(), lr=1e-4) # Typically, 1e-4 and 3e-4 work well for most problems


# Iterate through the batch
clip_15 = os.listdir(REMOTE_CLIP_DIR)

# Initialize embeddings
one_input_shape = [1, 768, 768]
att_mask_shape = [1, 768]
embed_shape = [1, 768]

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
        except Exception as e:
            print(f"Failed to load compressed numpy: {e}")
            continue
        object_list_of_str = []
        scene_seg_list = json.loads(scene_seg_list)
        
        # iterate over segments
        for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
            # print(np_loaded[f'arr_{segment_index}'].item()['captions'])
            frame_embedding       = np_loaded[f'arr_{segment_index}'].item()['frame_embeddings']
            caption_embedding     = np_loaded[f'arr_{segment_index}'].item()['text_caption_embeddings']
            whisper_text_captions = np_loaded[f'arr_{segment_index}'].item()['captions']
            
            frame_embedding       = torch.from_numpy(frame_embedding.reshape((768,))).to(device)
            caption_embedding     = torch.from_numpy(caption_embedding).to(device)

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
            decoder_input_embeds_arr = decoder_input_embeds_arr
            input_embeds_arr = input_embeds_arr
            attn_mask_arr = attn_mask_arr
            
            # print("Input shapes:", scene_embedding, caption_embedding, frame_embedding)
            labels = t5_tokenizer(whisper_text_captions, return_tensors="pt").input_ids.to(device)
            # outputs = t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, decoder_inputs_embeds=input_embeds_arr)
            outputs = t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, labels=labels)
            # outputs = t5.forward(inputs_embeds=input_embeds_arr, labels=labels)
            loss = outputs[0]
            wandb.log({"loss": loss})
            
            ''' backwards pass '''
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            train_itr += 1
            # print
            if train_itr % 500 == 0:
              print("Loss 👇👇👇")
              print(loss)
        
        # save checkpoints
        if train_itr % 500 == 0:
          t5.save_pretrained('v1_VPT_model')

t5.save_pretrained("BIG_PENIS_PREVAILS")