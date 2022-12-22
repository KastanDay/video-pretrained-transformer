import torch
import accelerate
from transformers import T5Tokenizer, T5EncoderModel, logging
import lovely_tensors as lt
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
    
  def encode(self, text):
    input_ids = self.tokenizer(text, return_tensors="pt").input_ids
    return self.model(input_ids=input_ids).last_hidden_state