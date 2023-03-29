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

# use enums?
# class torch_datatype(Enum):
#   fp16 = torch.float16
#   fp32 = torch.float32

# chosen_datatype = torch.float16
chosen_datatype = torch.float32
model_name = "google/flan-t5-small"


class FlanT5Encoder:

  def __init__(self, device: str = "cuda:1"):
    self.device = device if torch.cuda.is_available() else "cpu"
    print("In FlanT5Encoder", self.device)
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

  def encode_tvqa(self, sentence, max_tokens=804):
    # Tokenize the sentence and convert it to a PyTorch tensor
    tokens = self.tokenizer(sentence, return_tensors="pt", padding=False, truncation=False).to(self.device)

    # Generate the last hidden layer of the CLIP encoder
    lhs = self.model(**tokens).last_hidden_state

    return lhs.squeeze(0)
