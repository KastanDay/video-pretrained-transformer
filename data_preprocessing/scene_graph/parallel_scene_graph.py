import os
import time
import glob
import json
import jsonlines
from pathlib import Path
import numpy as np
import cv2

# our scene_graph code
from faster_OpenPSG.predict import Predictor 

parallel_dir      = 'parallel_15'
clip_input_dir    = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{parallel_dir}_clip_output/'
scene_output_path = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{parallel_dir}_scene_output.jsonl'
NUM_RELATIONS     = 10

# init model
my_pred = Predictor()
my_pred.setup()

start_time = time.monotonic()
# iterate over videos
for clip_npz_path in glob.glob(os.path.join(clip_input_dir, '*'), recursive = True):
  np_loaded = np.load(clip_npz_path, allow_pickle=True)
  object_list_of_str = []
  
  # iterate over segments
  for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
    # print(np_loaded[f'arr_{segment_index}'].item()['captions'])
    frame = np_loaded[f'arr_{segment_index}'].item()['segment_frames']
    frame = frame.reshape(336, 336, 3) # todo: ideally, get this from the shape property.
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
    curr_sro = my_pred.predict(image=frame_rgb, num_rel=NUM_RELATIONS)
    object_list_of_str.append(", ".join(curr_sro))
    for rel in curr_sro:
      if 'kissing' in rel:
        print(curr_sro)
    
  ''' Save the video stem to the output file. '''
  with jsonlines.open(scene_output_path, mode = 'a') as writer:
      writer.write(json.dumps(object_list_of_str))

  print(f"⏰ Ran scene graph {len(object_list_of_str)} frames in {(time.monotonic()-start_time):2f} seconds. Output to {scene_output_path}")