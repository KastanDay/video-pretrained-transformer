import os
import pprint
from enum import Enum

import accelerate
import lovely_tensors as lt
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer, logging

lt.monkey_patch()

# suppress: Some weights of the model checkpoint at google/flan-t5-large were not used when initializing model.
# This is expected because we're initializing the encoder-only. So the decoder weights are not used.
logging.set_verbosity_error()

# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # prevents ray from killing the process when it runs out of memory
os.environ['TRANSFORMERS_CACHE'] = '/mnt/teton/utils/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/teton/utils/cache/datasets'

# chosen_datatype = torch.float16
chosen_datatype = torch.float32
model_name = "google/flan-t5-large"


class FlanT5Encoder:

  def __init__(self, device: str = "cuda:1"):
    self.device = torch.device('cuda:0')  # if torch.cuda.is_available() else "cpu"
    print("In FlanT5Encoder, using:", self.device)
    self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    self.model = T5EncoderModel.from_pretrained(model_name, torch_dtype=chosen_datatype).to(self.device)
    # self.model = T5EncoderModel.from_pretrained(
    #     "google/flan-t5-large",
    #     # set device_map to only cuda:0
    #     # device_map="sequential",
    #     torch_dtype=torch.float16).to(device)  # I would really like to use this, but I think it's not supported by numpy or something.
    # self.model = T5EncoderModel.from_pretrained("google/flan-t5-large",
    #                                             device_map="auto",
    #                                             torch_dtype=torch.float16,
    #                                             max_memory={
    #                                                 0: "24GiB",
    #                                                 1: "0GiB",
    #                                             })

  def encode(self, input_dict):  #batch):
    """
    OLD param batch: list of {'db_index': int, 'caption': str}
    param input_dict: dict of {'db_index': int, 'caption': str}. batch_size is 1
    return: list of np.arrays, each of different shape [NUM_TOKENS, 1024]

    Important to pad to "max_length" so we can stack the inputs.
    keeping truncation=False for now because we really don't expect to go over length (with 15-word sequences), and I want to see errors if we do.
    """
    last_hidden_states_batch = []
    # for input_dict in batch:
    with torch.inference_mode():
      # print("ABOUT TO ENCODE CAPTION: ", input_dict["caption"])
      tokens = self.tokenizer(input_dict["caption"], return_tensors="pt", padding=False, truncation=False).to(self.device)
      # print("TOKENS: ", tokens)
      lhs = self.model(**tokens).last_hidden_state

      # CAST FROM 32 to 16 bit via .half() !!
      lhs = lhs.half().detach().cpu().numpy().reshape(-1, 1024)
      # print("LAST HIDDEN STATE: ", lhs)
      last_hidden_states_batch.append({'last_hidden_states': lhs, 'db_index': input_dict['db_index']})
    # return: list of np.arrays, each of different shape [NUM_TOKENS, 1024]
    return last_hidden_states_batch

  def encode_tvqa(self, sentence, truncate_shape=804):

    def pad_or_truncate_tensor(tensor):
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

    try:
      # Tokenize the sentence and convert it to a PyTorch tensor
      tokens = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True).to(self.device)

      # Generate the last hidden layer of the CLIP encoder
      with torch.inference_mode():
        lhs = self.model(**tokens).last_hidden_state

      # Truncate or pad last hidden states
      truncated_states = pad_or_truncate_tensor(lhs.squeeze(0))
      truncated_states = truncated_states.cpu()
      new_tensor = truncated_states
      if model_name == "google/flan-t5-small":
        new_tensor = torch.full((truncated_states.shape[0], 1024), -100)
        # Copy the original tensor's values to the first 512 columns of the new tensor
        new_tensor[:, :512] = truncated_states

      # print("------------------- ✅ SUCCESSFUL TEXT ENCODE (below here) -----------------------------")
      # print()
      # print("Sentence: ", sentence)
      # print()
      # tokens = self.tokenizer(sentence, return_tensors="pt", padding=False, truncation=False).input_ids
      # print("Tokens: ", tokens.shape)
      # print()
      # print("-------------------------------------------------------------------------------------")
      # torch.cuda.empty_cache()
      # print(torch.cuda.memory_stats())

      return new_tensor
    except Exception as e:
      print("------------------- ❌ FAILED TEXT ENCODE (below here) -----------------------------")
      print()
      print("Sentence: ", sentence)
      print()
      tokens = self.tokenizer(sentence, return_tensors="pt", padding=False, truncation=False).input_ids
      print("Tokens: ", tokens.shape)
      print()
      print("-------------------------------------------------------------------------------------")
      torch.cuda.empty_cache()
      # print(torch.cuda.memory_stats())
