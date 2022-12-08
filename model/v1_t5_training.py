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

BATCH_NAME          = "parallel_15"
MODEL_VERSION_NAME  = 'yt_pretrain'
BASE_DIR            = '/scratch/bbki/kastanday/whisper'
MODEL_SAVE_PATH     = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
REMOTE_WHISPER_FILE = f'{BASE_DIR}/{BATCH_NAME}_whisper_output.jsonl'
REMOTE_CLIP_DIR     = f'{BASE_DIR}/{BATCH_NAME}_clip_output'
REMOTE_SCENE_FILE   = f'{BASE_DIR}/{BATCH_NAME}_scene_output.jsonl'

# hyperparams 
learning_rate = 1e-4

wandb.config = {'learning_rate'     : learning_rate,
                'batch_name'        : BATCH_NAME,
                'model_save_path'   : MODEL_SAVE_PATH,
                }

# Instantiate clip
import clip
import torch
MODEL_SIZE = 'ViT-L/14@336px'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.  
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
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
t5 = T5ForConditionalGeneration.from_pretrained(MODEL_SAVE_PATH, torch_dtype=torch.float32, low_cpu_mem_usage=False).to(device) # float16, True
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base", return_special_tokens_mask=True)
# low_cpu_mem_usage(bool, optional) — Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. experimental.
optimizer = torch.optim.Adam(params =  t5.parameters(), lr=learning_rate) # Typically, 1e-4 and 3e-4 work well for most problems


def log_gradient_norm():
    try: 
        total_norm = 0
        parameters = [p for p in t5.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        wandb.log({"total_gradient_norm": total_norm})
        return total_norm
    except Exception as e:
        print("Failed to log gradient norm: ", e)

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
            log_gradient_norm()
            
            ''' backwards pass '''
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            train_itr += 1
        
        # save checkpoints
        if train_itr % 1000 == 0:
            print("SAVING MODEL CHECKPOING TO: ", f"{MODEL_VERSION_NAME}_iter{train_itr}")
            MODEL_SAVE_PATH = f"{MODEL_VERSION_NAME}_iter{train_itr}"
            t5.save_pretrained(MODEL_SAVE_PATH)


print(f"✅ Finished_training_batch_{BATCH_NAME}")
t5.save_pretrained(f"{BASE_DIR}/MODEL_CHECKPOINTS/Finished_training_batch_{BATCH_NAME}")