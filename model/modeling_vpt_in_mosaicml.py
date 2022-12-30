# Instantiate CustomT5
import os

import torch
import wandb
from composer.models import ComposerModel
from termcolor import colored
from transformers import (AutoModelWithLMHead, T5ForConditionalGeneration, T5Tokenizer)


class VPT_model(ComposerModel):
  '''
  Custom VPT implementaiton for MosaicML.
  https://docs.mosaicml.com/en/v0.11.1/trainer/using_the_trainer.html
  '''

  # todo: add a way to save the model
  # todo:
  def __init__(self,
               model_version_name: str = '',
               model_save_path: str = '',
               model_huggingface_name: str = "google/t5-v1_1-large"):
    super().__init__()

    # self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.device = "cpu"
    self.model = T5ForConditionalGeneration.from_pretrained(model_huggingface_name,
                                                            torch_dtype=torch.float32,
                                                            low_cpu_mem_usage=False).to(self.device)

    # todo: NEED TO GET THE LATEST CHECPOINT......
    if False and model_save_path and os.path.exists(model_save_path):
      # resume from checkpoint
      print("🚨 MAJOR WARNING: Probably loading the WRONG CHECKPOINT!!! ")
      # todo: use MODEL_SAVE_PATH and find the one with the highest iteration
      self.model = T5ForConditionalGeneration.from_pretrained(model_save_path,
                                                              torch_dtype=torch.float32,
                                                              low_cpu_mem_usage=False).to(self.device)  # float16, True
      # wandb.config.update({"starting_from_checkpoint": 0.1, "channels": 16})
    else:
      self.model = T5ForConditionalGeneration.from_pretrained(model_huggingface_name,
                                                              torch_dtype=torch.float32,
                                                              low_cpu_mem_usage=False).to(self.device)  # float16, True
    self.t5_tokenizer = T5Tokenizer.from_pretrained(model_huggingface_name, return_special_tokens_mask=True)

    self.model.train()
    self.train_itr = 0

    # wandb.init()

  def forward(self, batch):
    print("batch: ", batch)
    batch_size = batch['labels'].shape[0]
    input_embeds_arr = torch.cat([
        batch['clip_pooled_embedding'].reshape(batch_size, 1, -1), batch['clip_last_hidden_states'],
        batch['caption_embedding']
    ],
                                 dim=1)  # concat along sequence dimension
    print("Shape of input_embeds_arr: ", input_embeds_arr.shape)
    self.train_itr += 1
    # outputs = t5.forward()
    return self.model(inputs_embeds=input_embeds_arr, attention_mask=batch['attn_mask_arr'], labels=batch['labels'])

  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    todo: Placeholder so that training doesn't crash.
    '''
    return -1

  def loss(self, outputs, batch):
    # _, targets = batch
    # TODO Loss is from huggingface model outputs
    # wandb.log({"loss": outputs[0]})
    return outputs[0]


def log_gradient_norm():
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