import torch
import clip
import os
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Model, T5Config, AutoModelWithLMHead
import accelerate
import wandb
import lovely_tensors as lt
import math
from PIL import Image
lt.monkey_patch()
# !wandb login  -- reactivate later
device = 'cuda' if torch.cuda.is_available() else 'cpu'