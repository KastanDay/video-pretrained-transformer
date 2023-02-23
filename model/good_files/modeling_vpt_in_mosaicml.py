# Instantiate CustomT5
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import wandb
from composer.metrics import LanguageCrossEntropy  # , LanguagePerplexity
from composer.models import ComposerModel, HuggingFaceModel
from datasets import load_metric
from termcolor import colored
from torchmetrics import Metric, MetricCollection
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer)


class VPT_model(ComposerModel):
  '''
  Custom VPT implementaiton for MosaicML.
  https://docs.mosaicml.com/en/v0.11.1/trainer/using_the_trainer.html
  '''

  def __init__(self, model_version_name: str = '', model_save_path: str = '', model_huggingface_name: str = "google/t5-v1_1-large"):
    super().__init__()

    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_huggingface_name)  #.to(self.device)
    self.t5_tokenizer = AutoTokenizer.from_pretrained(model_huggingface_name, return_special_tokens_mask=True)
    # self.model.train()  -- already done by compser I think.

    # I got this vocab size from outputs.logits (it's rounded up from tokenizer.vocab_size)
    self.train_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)
    self.val_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)

  def forward(self, batch):
    input_embeds_arr = torch.cat([batch['clip_pooled_embedding'], batch['clip_last_hidden_states'], batch['caption_embedding']],
                                 dim=1)  # concat along sequence dimension
    return self.model.forward(inputs_embeds=input_embeds_arr, attention_mask=batch['attn_mask_arr'], labels=batch['labels'])

  # todo: UNTESTED
  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    todo: Placeholder so that training doesn't crash.
    '''
    ground_truth_labels = batch['labels'][0][batch['labels'][0] != -100]
    num_new_tokens = ground_truth_labels.shape[0]

    input_embeds_arr = torch.cat([batch['clip_pooled_embedding'], batch['clip_last_hidden_states'], batch['caption_embedding']],
                                 dim=1)  # concat along sequence dimension

    self.model.eval()
    with torch.no_grad():
      outputs = self.model.forward(inputs_embeds=input_embeds_arr,
                                   attention_mask=batch['attn_mask_arr'],
                                   labels=batch['labels'],
                                   return_dict=True)

    # generated vs actual tokens.
    # todo: jus tpass logits and labels, no edits at all. Works cuz I already set Ignore index = -100, so they match.
    self.val_cross_entropy.update(outputs.logits[0][0:num_new_tokens], batch['labels'][0][0:num_new_tokens])

    print("Cross entropy: ", self.val_cross_entropy.compute())
    wandb.log({"val_loss_cross_entropy": self.val_cross_entropy.compute()})

    # batch['labels'][0][0:num_new_tokens]

    return outputs

  def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
    return metric.update(outputs.logits, batch['labels'])

  # todo: UNTESTED
  def get_metrics(self, is_train: bool) -> Dict[str, Metric]:
    if is_train:
      return {"loss": self.train_cross_entropy}
    else:
      return {"val_loss": self.val_cross_entropy}

  def loss(self, outputs, batch):
    '''
        Return loss from huggingface model outputs.
        Docs'''
    loss = outputs[0].sum()
    wandb.log({"train_loss": loss})
    return loss

  # todo: not necessary?
  # def metrics(self, train: bool = False):
  #     return self.train_cross_entropy if train else self.val_cross_entropy
  # MetricCollection([self.val_acc, self.val_loss]) <-- for multiple metrics


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


def vpt_transform_dataset_to_batch(segment_batch, tokenizer_name: str):
  '''
  param: segment_batch: IterableOrderedDict. 1 SEGMENT (not batch_size, just one).
  The dataloader controls batch size, this just returns a single batch.
  
  caption embedding: keep only the first HALF of caption embedding. 
  label:             only keep 2nd half of caption to use as labels.
  clip_last_hidden_states: All clip hidden states.
  clip_pooled_embedding: 
  
  

  returns: batch dictionary.  Keys: input_embeds_arr, attn_mask_arr, labels_tokenized
                              Values: batched Torch Tensors of shape <1, 1024, 1024>. These are stacked to create a batch.
  '''
  # todo: remove all mention of cuda from this functino (no .todevice), then use pin=True.
  # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  # print("👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉👉 SEGMENT BATCH", flush=True)
  batch = {}  # keys: input_embeds_arr, attn_mask_arr, labels

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, return_special_tokens_mask=True)

  # maybe don't loop here??? just do it once. Not sure how it works on ds.batch_size setting.
  # Loop over BATCH_SIZE. Create dictionary where key = name, value = batched tensor
  for key, numpy_array in zip(segment_batch.keys(), segment_batch):
    # print("------------- PRINTING SEGMENT --------- ")
    if key == 'clip_pooled_embedding':
      # print("⭐️1️⃣ pooled embedding")
      numpy_array = numpy_array.reshape(1, -1)  # for batch size purposes.
      if key in batch.keys():
        batch[key] = torch.cat((batch[key], torch.from_numpy(numpy_array)), dim=0)
      else:
        batch[key] = torch.from_numpy(numpy_array)

    elif key == 'caption_embedding':
      # print("⭐️2️⃣ caption embedding")
      # keep only the first HALF of caption embedding.
      caption_length = numpy_array.shape[0]
      s_half = caption_length // 2
      # constant length of 446, pad with zeros. 446 is the max length of a caption (1024 - 577 - 1).
      caption_embedding_full_length = torch.zeros((446, 1024))
      caption_embedding_full_length[0:s_half] = torch.from_numpy(numpy_array[0:s_half])

      # setup attention mask now that we know full length of caption
      att_mask_shape = [1024]
      attn_mask_arr = torch.zeros(att_mask_shape)
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
        batch[key] = torch.cat((batch[key], torch.from_numpy(numpy_array)), dim=0)
      else:
        batch[key] = torch.from_numpy(numpy_array)

    # todo; clean this up IDK what exactly the labels look like.
    elif key == 'caption':
      caption = numpy_array[0]  # passed in as a single-element list.
      # print("⭐️4️⃣ CAPTION")
      # print(caption)
      full_caption_tokenized = tokenizer(caption, padding=False, truncation=True, return_tensors="pt").input_ids
      caption_length = full_caption_tokenized.shape[1]
      s_half = caption_length // 2
      # only keep 2nd half of caption to use as labels.
      proper_shape = full_caption_tokenized[0][s_half:].shape[0]
      labels_full_length = torch.ones((512), dtype=torch.int64) * -100
      labels_full_length[:proper_shape] = full_caption_tokenized[0][s_half:]  # 👈 take 2nd half!!
      if 'labels' in batch.keys():
        batch['labels'] = torch.cat((batch['labels'], labels_full_length), dim=0)
      else:
        batch['labels'] = labels_full_length
  return batch
