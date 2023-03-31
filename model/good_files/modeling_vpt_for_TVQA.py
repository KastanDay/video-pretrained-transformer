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
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          T5ForConditionalGeneration, T5Tokenizer)


class VPT_model(ComposerModel):
  '''
  Custom VPT implementaiton for MosaicML.
  https://docs.mosaicml.com/en/v0.11.1/trainer/using_the_trainer.html
  '''

  def __init__(self, model_huggingface_name: str = "google/t5-v1_1-large"):
    super().__init__()

    self.huggingface_model_name = model_huggingface_name

    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.huggingface_model_name)  #.to(self.device)
    self.t5_tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name, return_special_tokens_mask=True)
    # self.model.train()  -- already done by compser I think.

    # I got this vocab size from outputs.logits (it's rounded up from tokenizer.vocab_size)
    # self.train_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)
    self.val_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)
    self.yes_token_index = 4273
    self.no_token_index = 150


  def forward(self, batch):
    # dim=1 means concat along sequence dimension
    # input_embeds_arr = torch.cat([batch['context_vector'], batch['clip_last_hidden_states'], batch['caption_embedding']], dim=1)

    # todo: make attention mask array where tensors are -100.
    return self.model.forward(inputs_embeds=batch['context_vector'], attention_mask=batch['attn_mask_arr'], label=batch['label'])

  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    
    Todo: re-write this to use a pre-computed validation set (yt-1b-val). Use same pre-processing as training.
    
    TODO: re-write this for TVQA, pull out the "yes" token probability.
    If we set batch size to 5, then we can do 1 question per batch. 
    Makes it easier to report validation results.
    '''
    yes_no_label = batch['label']

    num_new_tokens = 1  # only output 1 token, hopefully yes or no.

    # input_embeds_arr = torch.cat([batch['clip_pooled_embedding'], batch['clip_last_hidden_states'], batch['caption_embedding']],
                                #  dim=1)  # concat along sequence dimension

    input_embeds_arr = batch["context_vector"]
    self.model.eval()
    with torch.no_grad():
      outputs = self.model.forward(inputs_embeds=batch['context_vector'],
                                   attention_mask=batch['attn_mask_arr'],
                                   labels=batch['label'],
                                   return_dict=True)

    ground_truth = torch.zeros_like(batch['context_vector'])
    yes_no_label = batch['label']
    if yes_no_label == 'yes':
      ground_truth[self.yes_token_index] = 1
    elif yes_no_label == 'no':
      ground_truth[self.no_token_index] = 1
    else:
      raise ValueError("yes_no_label must be 'yes' or 'no'")

    self.update_metric(outputs=outputs, ground_truth_onehot=ground_truth, metric=self.val_cross_entropy)
    self.val_cross_entropy.update(outputs, ground_truth)

    # print("Cross entropy: ", self.val_cross_entropy.compute())
    wandb.log({"val_loss_cross_entropy": self.val_cross_entropy.compute()})

    return outputs

  def update_metric(self, outputs: Any, ground_truth_onehot: Any, metric: Metric) -> None:
    return metric.update(outputs.logits, ground_truth_onehot)

  # todo: UNTESTED
  def get_metrics(self, is_train: bool) -> Dict[str, Metric]:
    if is_train:
      return {"train_loss_cross_entropy": self.train_cross_entropy}
    else:
      return {"val_loss_cross_entropy": self.val_cross_entropy}

  def loss(self, outputs, batch):
    '''
    Return loss from huggingface model outputs.
    Docs
    '''
    loss = outputs[0].sum()
    wandb.log({"train_loss": loss})
    return loss

  # todo: not necessary?
  # def metrics(self, train: bool = False):
  #     return self.train_cross_entropy if train else self.val_cross_entropy
  # MetricCollection([self.val_acc, self.val_loss]) <-- for multiple metrics

  def vpt_transform_dataset_to_batch(self, deeplake_batch):
    '''
    param: deeplake_batch: IterableOrderedDict. 1 SEGMENT (not batch_size, just one).
    The dataloader controls batch size, this just returns a single batch.
    
    context_vector:   a whole 1024x1024 context vector, prepared before. Right now it's CLIP and caption concatenated.
    label:            yes/no plaintext.

    returns: batch dictionary.  Keys: context_vector, attn_mask_arr, label
                                Values: batched Torch Tensors of shape <1, 1024, 1024>. These are stacked to create a batch.
    '''
    context_vector = torch.from_numpy(deeplake_batch['context_vector'])
    attn_mask_arr = torch.zeros_like(context_vector)
    # set attention mask to 0 wherever deeplake_batch['context_vector'] == -100
    attn_mask_arr[deeplake_batch['context_vector'] != -100] = 1

    tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name, return_special_tokens_mask=True)
    label_tokenized = tokenizer(deeplake_batch['label'], padding=True, truncation=True, return_tensors="pt").input_ids

    return {'context_vector': context_vector, 'attn_mask_arr': attn_mask_arr, 'label': label_tokenized}
