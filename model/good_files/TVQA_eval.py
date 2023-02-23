import json
from typing import Dict, List

import numpy as np


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

    model_name = 'google/flan-t5-XXL'
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

    subtitle = self.get_subtitle_from_clip(qa['vid_name'], float(qa['ts'].split('-')[0]), float(qa['ts'].split('-')[-1]))

    return [
        f"Some context: {subtitle}. Question: {qa['q']} Is it '{ans_candidate}'?"
        for ans_candidate in [qa['a0'], qa['a1'], qa['a2'], qa['a3'], qa['a4']]
    ]

  def run_prompts_get_best_answer(self, prompts: List[str]):
    all_yes_scores = []

    for prompt in prompts:
      # print(prompt)
      inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  # Batch size 1
      outputs = self.model.generate(**inputs, return_dict_in_generate=True, output_scores=True, early_stopping=True, max_new_tokens=2)
      # print(outputs)
      all_yes_scores.append(outputs.scores[0][0][4273])
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
