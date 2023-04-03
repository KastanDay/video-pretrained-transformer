import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5Config, AutoModelWithLMHead
import accelerate
import wandb
# !wandb login  -- reactivate later
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
MODEL SELECTION

T5 V1.1 --  https://huggingface.co/docs/transformers/model_doc/t5v1.1 && https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#t511
small - base - large - 3b/xl - 11b/xxl

OG: t5-small

'google/t5-base-lm-adapt' # largest on my server (without float16)
'google/t5-xl-lm-adapt'
'''

# MODEL_SIZE = "t5-base"
MODEL_NAME = "google/t5-small-lm-adapt"
# config = T5Config.from_pretrained(MODEL_NAME)
t5 = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, return_special_tokens_mask=True)
# low_cpu_mem_usage(bool, optional) — Tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model. experimental.


''' PREP EMBEDDING INPUTS '''

input_embeds_arr = np.random.rand( 6, 512, 512 )
decoder_input_embeds_arr = np.random.rand( 6, 512, 512 )
attn_mask_arr = np.ones( (6, 512, 512) )
# input_embeds_arr = np.random.rand( 6, 768, 768 )
# decoder_input_embeds_arr = np.random.rand( 6, 768, 768 )
# attn_mask_arr = np.ones( (6, 768, 768) )
print(input_embeds_arr.shape)
print(attn_mask_arr.shape)
print(type(input_embeds_arr[0,0,0]))

# convert from fp64 to fp32
input_embeds_arr = np.ndarray.astype(input_embeds_arr, dtype=np.float32)
attn_mask_arr = np.ndarray.astype(attn_mask_arr, dtype=np.float32)

# input_embeds_arr as a toch tensor
input_embeds_arr = torch.from_numpy(input_embeds_arr).to(device)
attn_mask_arr = torch.from_numpy(attn_mask_arr).to(device)

# decoder_inputs = tokenize('This is the target sentence.')
import torch.nn.functional as F
decoder_input_ids = tokenizer("This is the target output sentence, aka the video caption. I like tacos because they are so delicious.", return_tensors="pt").input_ids.to(device)
decoder_input_ids = F.pad(decoder_input_ids, (0, 512-decoder_input_ids.shape[1]), value=tokenizer.pad_token_id)
print("Decoder_input_ids", decoder_input_ids.shape)

''' forward pass '''
t5.train()
# outputs = t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, decoder_inputs_embeds=input_embeds_arr)
outputs = t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, decoder_input_ids=decoder_input_ids)
loss = outputs[0]
print("loss", loss.shape)

''' backwards pass '''
optimizer = torch.optim.Adam(params =  t5.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.sum().backward()
optimizer.step()

print("✅ SUCCESS ✅")