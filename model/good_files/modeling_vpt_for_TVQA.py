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

  def __init__(self, model_huggingface_name: str = "google/t5-v1_1-large"):
    super().__init__()

    self.huggingface_model_name = model_huggingface_name

    self.model = AutoModelForSeq2SeqLM.from_pretrained(self.huggingface_model_name)  #.to(self.device)
    self.t5_tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name, return_special_tokens_mask=True)
    # self.model.train()  -- already done by compser I think.

    # I got this vocab size from outputs.logits (it's rounded up from tokenizer.vocab_size)
    self.vocab_size = 32128
    self.train_cross_entropy = LanguageCrossEntropy(self.vocab_size, ignore_index=-100)
    self.val_cross_entropy = LanguageCrossEntropy(self.vocab_size, ignore_index=-100)


  def forward(self, batch):
    # dim=1 means concat along sequence dimension
    # input_embeds_arr = torch.cat([batch['context_vector'], batch['clip_last_hidden_states'], batch['caption_embedding']], dim=1)

    # todo: make attention mask array where tensors are -100.
    print("In modeling forward()")
    print('batch[context_vector].shape', batch['context_vector'].shape, flush=True)
    print('batch[attn_mask_arr].shape', batch['attn_mask_arr'].shape, flush=True)
    print('batch[label].shape', batch['label'].shape, flush=True)

    c = batch['context_vector']  #.reshape(1024, 1024)
    a = batch['attn_mask_arr']  #.reshape(1024, 1024)
    l = batch['label']  #.reshape(1, 2)
    return self.model.forward(inputs_embeds=c, attention_mask=a, labels=l)

  def eval_forward(self, batch, outputs=None):
    '''
    Docs: https://docs.mosaicml.com/en/v0.11.1/api_reference/generated/composer.ComposerModel.html#composer.ComposerModel.eval_forward
    '''
    yes_no_label = batch['label']

    # todo can we set num_new_tokens in forward()? No right?? Maybe.
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

    for metric in self.get_metrics(is_train=False).values():
      self.update_metric(outputs=outputs, batch=batch, metric=metric)
    wandb.log({"val_loss_cross_entropy": self.val_cross_entropy.compute()})

    return outputs

  def update_metric(self, outputs: Any, batch: Any, metric: Metric) -> None:
    """ 
    Uniform interface for all metrics.
    Docs: https://docs.mosaicml.com/en/latest/trainer/using_the_trainer.html#training-loop
    """
    if isinstance(metric, LanguageCrossEntropy):

      # set
      ground_truth_matrix = torch.zeros((len(batch), self.vocab_size))
      yes_no_label = batch['label']  # tensor of [[105, 1]] due to stop token.

      # want a mask of [batch, vocab_size] where 1 at yes/no index, 0 elsewhere
      # make a scatter mask?

      for i, elm in enumerate(batch):
        no_token_index = self.t5_tokenizer.convert_tokens_to_ids('no')
        yes_token_index = self.t5_tokenizer.convert_tokens_to_ids('yes')

        if no_token_index in elm['label']:
          ground_truth_matrix[i, no_token_index] = 1
        elif yes_no_label in elm['label']:
          ground_truth_matrix[i, yes_token_index] = 1
        else:
          raise ValueError("yes_no_label must be 'yes' or 'no'")

      return metric.update(outputs.logits, ground_truth_matrix)
    else:
      raise NotImplementedError("Only LanguageCrossEntropy is implemented.")
    return metric.update(outputs.logits, batch)

  def get_metrics(self, is_train: bool) -> MetricCollection:
    """ 
    Uniform interface for all metrics.
    
    MetricCollection for multiple metrics.
    Docs: https://docs.mosaicml.com/en/latest/trainer/using_the_trainer.html#training-loop
    
    """
    if is_train:
      return MetricCollection([self.train_cross_entropy])
    else:
      return MetricCollection([self.val_cross_entropy])

  def loss(self, outputs, batch):
    '''
    Return loss from huggingface model outputs.
    Docs
    '''
    loss = outputs[0].sum()
    wandb.log({"train_loss": loss})
    return loss

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
    attn_mask_arr[context_vector != -100] = 1

    tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_name, return_special_tokens_mask=True)
    label_tokenized = tokenizer(deeplake_batch['label'][0], padding=True, truncation=True, return_tensors="pt").input_ids

    # ONE OF THESE IS A LIST when it shouldn’t be.

    one_batch_dict = {'context_vector': context_vector, 'attn_mask_arr': attn_mask_arr, 'label': label_tokenized}
    print("----------------- ONE BATCH BELOW HERE -----------------")
    print("context_vector:", context_vector)
    print("context_vector type:", type(context_vector))
    print("attn_mask_arr:", attn_mask_arr)
    print("attn_mask_arr:", type(attn_mask_arr))
    print("Label Tokenized:", label_tokenized)
    print("Label Tokenized type:", type(label_tokenized))
    print("-------------------------------------------------")

    return one_batch_dict
