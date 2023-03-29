import json
import os
import pathlib
import sys
from typing import Any, Dict, List

import more_itertools
import numpy as np
import pandas as pd
import torch
from PIL import Image

os.environ['TRANSFORMERS_CACHE'] = '/mnt/teton/utils/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/teton/utils/cache/datasets'

sys.path.append("../../data_preprocessing/parallel_processing")
from clip_encoder import ClipEncoder
from text_encoder import FlanT5Encoder


class TVQA_eval():

  def __init__(self):

    train_filepath = "/mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_qa_release/tvqa_train.jsonl"
    with open(train_filepath, 'r') as f:
      self.train_qa_json = [json.loads(line) for line in f]
    # len(train_qa_json)  # 122,039

    subtitles_filepath = "/mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_preprocessed_subtitles.jsonl"
    with open(subtitles_filepath, 'r') as f:
      # self.subtitles = [json.loads(line) for line in f]
      self.subtitles = pd.read_json(subtitles_filepath, lines=True)
      self.subtitles = self.subtitles.set_index('vid_name')
    # len(subtitles)  # 21,793

    # instantiate expert models
    self.clip_encoder = ClipEncoder(debug=True)
    self.text_encoder = FlanT5Encoder()

    self.tvqa_train_to_path = {
        "House M.D.": 'house_frames',
        "The Big Bang Theory": 'bbt_frames',
        "Castle": 'castle_frames',
        "How I Met You Mother": 'met_frames',
        "Grey's Anatomy": 'grey_frames',
        "Friends": 'friends_frames',
    }

    self.vid_name_prefix_to_path = {
        "house": 'house_frames',
        "castle": 'castle_frames',
        "met": 'met_frames',
        "grey": 'grey_frames',
        "friends": 'friends_frames',
        "": 'bbt_frames',  # no prefix at all used for bbt, it's the "default"
    }

  # Deprecated
  def get_subtitle_from_clip(self, vid_name: str, ts: str):
    start_time, end_time = ts.split('-')
    subtitle = ''
    sub = self.subtitles.loc[vid_name]
    for text in sub['sub']:
      if text['start'] >= float(start_time) and text['end'] <= float(end_time):
        subtitle += text['text'] + ' '
    return subtitle.strip()

  def get_all_subtitles(self, vid_name: str):
    '''Returns a string with every subtitle from a given video'''
    sub = ' '.join([_dict["text"] for _dict in self.subtitles.loc[vid_name]["sub"]])
    return sub.strip()

  def qa_to_prompt(self, qa: Dict):
    '''
    TODO: update prompt to be VPT-compatible. 
    TODO: add </s> token to end of prompt... during training.
    '''
    # In this version, we only get the subtitles relevant to the time stamp
    # subtitle = self.get_subtitle_from_clip(qa['vid_name'], float(qa['ts'].split('-')[0]), float(qa['ts'].split('-')[-1]))
    # In this version, we get all subtitles from the video
    print(qa)
    print(qa['vid_name'])
    subtitle = self.get_all_subtitles(qa['vid_name'])

    # Should change this to all frames. Either way, this is irrelevant for this function
    # image_embed = self.get_clip_embed_from_vid_name(qa['vid_name'], float(qa['ts'].split('-')[0]), float(qa['ts'].split('-')[-1]))

    return [
        f"Context: {subtitle}. Question: {qa['q']} Is it '{ans_candidate}'?"
        for ans_candidate in [qa['a0'], qa['a1'], qa['a2'], qa['a3'], qa['a4']]
    ]

  def combine_modality_encodings(self, text_encoding, image_encoding):
    '''Untested btw'''
    num_text_embeddings, _ = text_encoding.shape
    num_image_embeddings, _ = image_encoding.shape

    assert num_image_embeddings + num_text_embeddings <= 1024, f"the given encoding has more than 1024 embeddings! Text is {num_text_embeddings} and image is {num_image_embeddings}"

    combined_tensor = torch.cat((text_encoding, image_encoding), dim=0)
    # Pad the tensor with -100 to make it a [1024, 1024] tensor

    # There shouldn't be any padding left
    # pad_size = (1024 - combined_tensor.shape[0], 1024)
    # padded_tensor = torch.nn.functional.pad(combined_tensor, pad_size, value=-100)

    # Reinsert 
    assert combined_tensor.shape[0] == 1024, f"the combined encoding does not have exactly 1024 embeddings! Dimension: {combined_tensor.shape}"

    return combined_tensor

  def create_context_vectors(self, question):
    '''Combine the two vectors to create the context vector'''
    try:
      all_context_vectors = []
      text_encodings = self.get_flant5_embed_from_vid_name(question)
      image_encoding = self.get_clip_embed_from_vid_name(question['vid_name'])  # this should be get_clip_embed_from_vid_name
      for text_encoding in text_encodings:
        all_context_vectors.append(self.combine_modality_encodings(text_encoding, image_encoding))
      return all_context_vectors
    except FileNotFoundError as e:
      print(e)
      print(f"WARNING: Could not find video {question['vid_name']}. Skipping...")

  def pad_or_truncate_tensor(self, tensor, truncate_shape):
      target_shape = [truncate_shape, 1024]
      tensor_shape = tensor.shape

      # If tensor shape is larger than the target shape, truncate the tensor
      if tensor_shape[0] > target_shape[0]:
          truncated_tensor = tensor[:target_shape[0], :]
          return truncated_tensor

      # If tensor shape is smaller than the target shape, pad the tensor
      elif tensor_shape[0] < target_shape[0]:
          padding_shape = (target_shape[0] - tensor_shape[0], target_shape[1])
          padded_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_shape[0]), value=-100)
          return padded_tensor

      # If tensor shape is already the target shape, return the tensor
      else:
          return tensor
      
  def vid_name_to_frames_path(self, vid_name):
    '''
    Example:
    vid_name: house_s02e05_seg02_clip_11
    
    show_name_path = house_frames
    clip_name_path = s02e05_seg02_clip_11
    
    returns: /mnt/teton/vpt/data/benchmark_datasets/TVQA/uncompressed_frames/frames_hq/house_frames/s02e05_seg02_clip_11
    '''

    show_name, clip_name_path = vid_name.split('_', 1)[0], vid_name.split('_', 1)[1]
    if show_name not in self.vid_name_prefix_to_path:
      show_name_path = 'bbt_frames'
      clip_name_path = vid_name  # no prefix at all used for bbt. vid_name is actually the 'clip_name_path'
    else:
      show_name_path = self.vid_name_prefix_to_path[show_name]

    frames_dir = os.path.join('/mnt/teton/vpt/data/benchmark_datasets/TVQA/uncompressed_frames/frames_hq', show_name_path, clip_name_path)
    if not os.path.exists(frames_dir):
      raise FileNotFoundError(f"frames_dir {frames_dir} does not exist")

    return frames_dir

  def get_clip_embed_from_vid_name(self, vid_name, max_encodings_to_make=220):
    '''
    ✅ Working
    
    PARAMS
    vid_name (from the subtitles jsonl file)
    max_encodings_to_make: int. It will only encode the FIRST max_encodings_to_make frames.
    
    RETURNS
    A list of clip embeddings for that video segment. There is a VARIABLE number of frames per "clip". 
    
    Notes: 
      * Castle avg frames/clip = 274.79
      * BBT avg frames/clip    = 186.38
    
    220 frames per clip is a good number to use.
    1024 (Flan-T5-XXL window size) - 220 = 804 text encodings to use. 
    '''
    # collect all frames from filepath
    segment_frames_paths = pathlib.Path(self.vid_name_to_frames_path(vid_name)).glob('*.jpg')
    frames_PIL_list = []
    for path in segment_frames_paths:
      if len(frames_PIL_list) >= max_encodings_to_make:
        break
      with Image.open(path) as img:
        frames_PIL_list.append(np.array(img))

    # split frames_PIL_list into batches of 100 (to avoid OOM)
    batch_list = list(more_itertools.batched(frames_PIL_list, 110))

    clip_embeddings = []
    for batch in batch_list:
      clip_embeddings.extend(self.clip_encoder.run_clip(batch, only_return_pooled_embeds=True))


    # Converting a list of tensors to a tensor
    # Get the shape of the first tensor in the list
    tensor_shape = clip_embeddings[0].shape
    # Create a tensor of zeros with the appropriate shape
    tensor = torch.zeros(len(clip_embeddings), *tensor_shape)
    # Fill the tensor with the values from the input list
    for i, t in enumerate(clip_embeddings):
        tensor[i, ...] = t
    fixed_tensor = self.pad_or_truncate_tensor(tensor, max_encodings_to_make)
    fixed_tensor.cpu()
    return fixed_tensor

  def get_flant5_embed_from_vid_name(self, question_sample, max_encodings_to_make=804):
    '''
    param: 
    question_sample is a dictionary with ['a0',  'a1',  'a2',  'a3', 'a4',    'answer_idx',   'q',   'qid',  'show_name', 'ts', 'vid_name'] as keys

    returns: list of last hidden state of encoding, (prompt, subtitles, answer) for each answer
    '''
    assert type(question_sample) == dict, f"question_sample must be a dictionary. It is a {type(question_sample)}."

    all_prompts = self.qa_to_prompt(question_sample)
    all_encodings = []
    for prompt in all_prompts:
      all_encodings.append(self.text_encoder.encode_tvqa(prompt, truncate_shape=max_encodings_to_make))
    return all_encodings

  def run_prompts_get_best_answer(self, prompts: List[str]):
    all_yes_scores = []

    with torch.no_grad():
      for prompt in prompts:
        # print(prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # Batch size 1
        outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True, early_stopping=True, max_new_tokens=2)
        # print(outputs)
        all_yes_scores.append(outputs.scores[0][0][4273].cpu())
        # print('yes_score =', outputs.scores[0][0][4273])
        # print('no_score =', outputs.scores[0][0][150])
        # print(self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False))

    return np.array(all_yes_scores).argmax()

  def accuracy(self, actual, predicted):
    correct = 0
    total = len(actual)
    for i in range(total):
      if actual[i] == predicted[i]:
        correct += 1
    return correct / total


def main():
  eval = TVQA_eval()

  # can iterate over all questions in train_qa_json.
  # print('\n'.join(eval.qa_to_prompt(eval.train_qa_json[0])))
  # print(eval.qa_to_prompt(eval.train_qa_json[0]))

  # get a baseline using Flan-T5.

  actual = []
  predicted = []
  for i in range(5):
    predicted_ans_idx = eval.run_prompts_get_best_answer(eval.qa_to_prompt(eval.train_qa_json[i]))
    actual.append(int(eval.train_qa_json[i]['answer_idx']))
    predicted.append(predicted_ans_idx)
    print("Predicted answer:", predicted_ans_idx)
    print("Actual answer:   ", eval.train_qa_json[i]['answer_idx'])

  print("Accuracy:", eval.accuracy(actual, predicted))


if __name__ == '__main__':
  main()
