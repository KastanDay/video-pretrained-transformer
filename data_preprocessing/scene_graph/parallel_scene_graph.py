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

parallel_dir      = 'parallel_17'
clip_input_dir    = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{parallel_dir}_clip_output/'
scene_output_path = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/{parallel_dir}_scene_output.jsonl'
NUM_RELATIONS     = 10

# init model
my_pred = Predictor()
my_pred.setup()

# load dataset
import deeplake
dataset_name = '/mnt/storage_ssd/FULL_v3_parallel_ingest_p15'
assert os.path.exists(dataset_name), print("Please provide the proper database path")
ds = deeplake.load(dataset_name)
print(ds.summary())
all_stems = ds.video_stem.data()
already_completed_stems = set(all_stems)
print("already completed: ", len(already_completed_stems))

try: 
  # segment_scene_graph_str = ds.create_tensor('segment_scene_graph_str', htype='str', sample_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html
  segment_scene_graph_str = ds.create_tensor('segment_scene_graph_str', htype='str', chunk_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html
except Exception as e:
  print(e)

start_time = time.monotonic()
# iterate over videos
for i, sample in enumerate(ds):
  # if not sample.segment_scene_graph_str:
  # todo: check if scene graph exists at this index, or not. Maybe length of that property? 
  frame = sample.segment_frames.numpy()
  curr_sro = my_pred.predict(image=frame, num_rel=NUM_RELATIONS)
  with ds:
    ds.segment_scene_graph_str.append(", ".join(curr_sro))
  
  # todo: check if append is okay, or if we need index
  
  # iterate over segments
  # for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
  #   # print(np_loaded[f'arr_{segment_index}'].item()['captions'])
  #   frame = np_loaded[f'arr_{segment_index}'].item()['segment_frames']
  #   frame = frame.reshape(336, 336, 3) # todo: ideally, get this from the shape property.
  #   frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
  #   curr_sro = my_pred.predict(image=frame_rgb, num_rel=NUM_RELATIONS)
  #   object_list_of_str.append(", ".join(curr_sro))
  #   for rel in curr_sro:
  #     if 'kissing' in rel:
  #       print(curr_sro)
    
  # ''' Save the video stem to the output file. '''
  # with jsonlines.open(scene_output_path, mode = 'a') as writer:
  #     writer.write(json.dumps(object_list_of_str))

  print(f"⏰ Ran scene graph {len(object_list_of_str)} frames in {(time.monotonic()-start_time):2f} seconds. Output to {scene_output_path}")