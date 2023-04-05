import torch
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5Config

config = T5Config.from_pretrained("t5-small")
t5 = T5ForConditionalGeneration(config=config)
tokenizer = T5Tokenizer.from_pretrained("t5-small", return_special_tokens_mask=True, mlm=False)

input_embeds_arr = np.random.rand( 2, 512, 512 )
decoder_input_embeds_arr = np.random.rand( 2, 512, 512 )
attn_mask_arr = np.ones( (2, 512, 512) )
print(input_embeds_arr.shape)
print(attn_mask_arr.shape)
print(type(input_embeds_arr[0,0,0]))

# convert from fp64 to fp32
input_embeds_arr = np.ndarray.astype(input_embeds_arr, dtype=np.float32)
attn_mask_arr = np.ndarray.astype(attn_mask_arr, dtype=np.float32)

# input_embeds_arr as a toch tensor
input_embeds_arr = torch.from_numpy(input_embeds_arr)
attn_mask_arr = torch.from_numpy(attn_mask_arr)


# Masked language modeling masking
# todo: ensure correct masking.
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.25, mlm=True, return_tensors="pt")
# print(data_collator)
res = data_collator(input_embeds_arr)

print(res)

# todo: run in forward pass
t5.forward(inputs_embeds=input_embeds_arr, attention_mask=attn_mask_arr, decoder_inputs_embeds=input_embeds_arr)

