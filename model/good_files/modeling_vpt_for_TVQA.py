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

    self.huggingface_model_name = model_huggingface_name

    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.huggingface_model_name)  #.to(self.device)
    self.t5_tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name, return_special_tokens_mask=True)
    # self.model.train()  -- already done by compser I think.

    # I got this vocab size from outputs.logits (it's rounded up from tokenizer.vocab_size)
    self.train_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)
    self.val_cross_entropy = LanguageCrossEntropy(vocab_size=32128, ignore_index=-100)

  def forward(self, batch):
    # dim=1 means concat along sequence dimension
    # input_embeds_arr = torch.cat([batch['context_vector'], batch['clip_last_hidden_states'], batch['caption_embedding']], dim=1)  
    
    # todo: make attention mask array where tensors are -100. 
    return self.model.forward(inputs_embeds=batch['context_vector'], attention_mask=batch['attn_mask_arr'], labels=batch['labels'])

  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    
    Todo: re-write this to use a pre-computed validation set (yt-1b-val). Use same pre-processing as training.
    
    TODO: re-write this for TVQA, pull out the "yes" token probability.
    If we set batch size to 5, then we can do 1 question per batch. 
    Makes it easier to report validation results.
    '''
    ground_truth_labels = batch['labels'][0][batch['labels'][0] != -100]
    num_new_tokens = .shape[0]

    input_embeds_arr = torch.cat([batch['clip_pooled_embedding'], batch['clip_last_hidden_states'], batch['caption_embedding']],
                                 dim=1)  # concat along sequence dimension

    self.model.eval()
    with torch.no_grad():
      outputs = self.model.forward(inputs_embeds=input_embeds_arr,
                                   attention_mask=batch['attn_mask_arr'],
                                   labels=batch['labels'],
                                   return_dict=True)

    # generated vs actual tokens.
    # todo: just pass logits and labels, no edits at all. Works cuz I already set Ignore index = -100, so they match.
    # self.val_cross_entropy.update(outputs.logits, batch['labels']) # this should work, is simpler, but untested
    self.val_cross_entropy.update(outputs.logits[0][0:num_new_tokens], batch['labels'][0][0:num_new_tokens])

    # print("Cross entropy: ", self.val_cross_entropy.compute())
    wandb.log({"val_loss_cross_entropy": self.val_cross_entropy.compute()})

    return outputs

  def update_metric(self, batch: Any, outputs: Any, metric: Metric) -> None:
    return metric.update(outputs.logits, batch['labels'])

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