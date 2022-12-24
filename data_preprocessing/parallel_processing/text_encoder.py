import torch
import accelerate
from transformers import T5Tokenizer, T5EncoderModel, logging
import lovely_tensors as lt
import numpy as np
import pprint
lt.monkey_patch()

# suppress: Some weights of the model checkpoint at google/flan-t5-large were not used when initializing model.
# This is expected because we're initializing the encoder-only. So the decoder weights are not used.
logging.set_verbosity_error() 

class FlanT5Encoder():
  def __init__(self):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    self.model = T5EncoderModel.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
  
  def encode(self, batch):
    '''
    param batch: list of {'db_index': int, 'caption': str}
    return: list of np.arrays, each of different shape [NUM_TOKENS, 1024]
    
    Important to pad to "max_length" so we can stack the inputs.
    keeping truncation=False for now because we really don't expect to go over length (with 15-word sequences), and I want to see errors if we do. 
    '''
    last_hidden_states_batch = []
    for input_dict in batch:
      with torch.inference_mode():
        tokens = self.tokenizer(input_dict['caption'], return_tensors="pt", padding=False, truncation=False)
        last_hidden_states_batch.append(self.model(**tokens).last_hidden_state.detach().cpu().numpy().reshape(-1, 1024))
    # return: list of np.arrays, each of different shape [NUM_TOKENS, 1024]
    return last_hidden_states_batch