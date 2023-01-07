# Instantiate CustomT5
import os

import torch
import wandb
from composer.models import ComposerModel, HuggingFaceModel
from termcolor import colored
from transformers import (AutoModelWithLMHead, T5ForConditionalGeneration, T5Tokenizer)


class VPT_model(ComposerModel):
  '''
  Custom VPT implementaiton for MosaicML.
  https://docs.mosaicml.com/en/v0.11.1/trainer/using_the_trainer.html
  '''

  def __init__(self,
               model_version_name: str = '',
               model_save_path: str = '',
               model_huggingface_name: str = "google/t5-v1_1-large"):
    super().__init__()

    self.model = T5ForConditionalGeneration.from_pretrained(model_huggingface_name)  #.to(self.device)
    self.model.train()

  def forward(self, batch):
    input_embeds_arr = torch.cat(
        [batch['clip_pooled_embedding'], batch['clip_last_hidden_states'], batch['caption_embedding']],
        dim=1)  # concat along sequence dimension
    return self.model.forward(inputs_embeds=input_embeds_arr,
                              attention_mask=batch['attn_mask_arr'],
                              labels=batch['labels'])

  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    todo: Placeholder so that training doesn't crash.
    '''
    return 0

  def loss(self, outputs, batch):
    '''
    Return loss from huggingface model outputs.
    '''
    return outputs[0].sum()


'''
def log_gradient_norm():
# practitioners recommend logging the average norm of the grad. 
  try:
    # start_time = time.monotonic()
    total_norm = 0
    # todo: make member function
    parameters = [p for p in t5.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
      param_norm = p.grad.detach().data.norm(2)
      total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    wandb.log({"total_gradient_norm": total_norm})
    # print(f"Time to log norm: {(time.monotonic() - start_time):2f} seconds") # 0.01 seconds
    return total_norm
  except Exception as e:
    print("Failed to log gradient norm: ", e)
'''