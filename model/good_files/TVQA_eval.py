import json
import os
from typing import Dict, List

import numpy as np
import torch

os.environ['TRANSFORMERS_CACHE'] = '/mnt/teton/utils/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/teton/utils/cache/datasets'


class TVQA_eval():

  def __init__(self):

    train_filepath = "/mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_qa_release/tvqa_train.jsonl"
    with open(train_filepath, 'r') as f:
      self.train_qa_json = [json.loads(line) for line in f]
    # len(train_qa_json)  # 122,039

    subtitles_filepath = "/mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_preprocessed_subtitles.jsonl"
    with open(subtitles_filepath, 'r') as f:
      self.subtitles = [json.loads(line) for line in f]
    # len(subtitles)  # 21,793

    # load model
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    self.device = 'cuda:0'

    model_name = 'google/flan-t5-xl'
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)

  def get_subtitle_from_clip(self, vid_name: str, start_time: float, end_time: float):
    subtitle = ''
    for sub in self.subtitles:
      if sub['vid_name'] == vid_name:
        for text in sub['sub']:
          if text['start'] >= float(start_time - 1) and text['end'] <= float(end_time + 1):
            subtitle += text['text'] + ' '
    return subtitle.strip()

  def qa_to_prompt(self, qa: Dict):
    '''
    TODO: update prompt to be VPT-compatible. 
    TODO: add </s> token to end of prompt... during training.
    '''

    subtitle = self.get_subtitle_from_clip(qa['vid_name'], float(qa['ts'].split('-')[0]), float(qa['ts'].split('-')[-1]))

    image_embed = self.get_clip_embed_from_vid_name(qa['vid_name'], float(qa['ts'].split('-')[0]), float(qa['ts'].split('-')[-1]))

    return [
        f"Context: {subtitle}. Question: {qa['q']} Is it '{ans_candidate}'?"
        for ans_candidate in [qa['a0'], qa['a1'], qa['a2'], qa['a3'], qa['a4']]
    ]

  def vid_name_to_path(self, show_name, vid_name):
    tvqa_train_to_path = {
        "House M.D.": 'house_frames',
        "The Big Bang Theory": 'bbt_frames',
        "Castle": 'castle_frames',
        "How I Met You Mother": 'met_frames',
        "Grey's Anatomy": 'grey_frames',
        "Friends": 'friends_frames',
    }

    frames_dir = os.path.join('/teton/vpt/data/benchmark_datasets/TVQA/uncompressed_frames/frames_hq', tvqa_train_to_path['show_name'],
                              vid_name)
    # todo need to pick some subset of frames to use inside this folder.
    # get all files in this folder
    frames = os.listdir(frames_dir)
    return frames

  def get_clip_embed_from_vid_name(self, vid_name, start_time, end_time):
    base_frames_filepath = "/teton/vpt/data/benchmark_datasets/TVQA/uncompressed_frames/frames_hq"
    base_frames_filepath

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
