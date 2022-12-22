import os
import time
import glob
import json
import jsonlines
from pathlib import Path
import numpy as np
# import cv2
from PIL import Image

# our scene_graph code
from faster_OpenPSG.predict import Predictor 

# TODO: way to track progress and restart. 

# parallel_dir      = 'parallel_16'
# clip_input_dir    = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{parallel_dir}_clip_output/'
img_path          = f'/home/kastan/thesis/video-pretrained-transformer/vqa/train2014'
scene_output_path = f'/home/kastan/thesis/video-pretrained-transformer/vqa/v2_train_scene_graph.jsonl'
NUM_RELATIONS     = 10

# init model
my_pred = Predictor()
my_pred.setup()

start_time = time.monotonic()
# iterate over videos
for img_path in glob.glob(os.path.join(img_path, '*'), recursive = True):
  # try:
  #   img = Image.open(img_path)
  # except Exception as e:
  #   print(f"Failed to load images: {e}")
  #   continue
  try:
    curr_sro = my_pred.predict(image=img_path, num_rel=NUM_RELATIONS)
    curr_sro = ", ".join(curr_sro) # list --> string
  except Exception as e:
    print(f"failed scene graph predict: {e}")
    continue
  
  result_json = json.dumps({
    'input_img_path': img_path,
    'scene_graph_string': curr_sro
  })
  
  with jsonlines.open(scene_output_path, mode = 'a') as writer:
      writer.write(json.dumps(result_json))
  
  print(f"⏰ Ran scene graph on 1 frame in {(time.monotonic()-start_time):2f} seconds. Output to {scene_output_path}")