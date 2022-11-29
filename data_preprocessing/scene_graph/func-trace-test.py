# import functiontrace
# functiontrace.trace()

from faster_OpenPSG.predict import Predictor 
my_pred = Predictor()
my_pred.setup()

import numpy as np
import cv2
from PIL import Image
import time
import glob

# except KeyError as e:
  # print("End of file")

# glob.glob()
NUM_RELATIONS = 1

path = '/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_15_clip_output/J-DG5O7gqdY_All4TubeVLOGS_15315_This Is Why We Had To Move To Florida.npz'

start_time = time.monotonic()

# they're always named arr_0 thru num_segments 
np_loaded = np.load(path, allow_pickle=True)
object_list_of_str = []
for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
  # print(np_loaded[f'arr_{segment_index}'].item()['captions'])
  frame = np_loaded[f'arr_{segment_index}'].item()['segment_frames']
  frame = frame.reshape(336, 336, 3)
  frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  
  # display(Image.fromarray(frame_rgb))
  # print('loaded: ')
  # print(frame_rgb.shape)
  
  curr_sro = my_pred.predict(image=frame_rgb, num_rel=NUM_RELATIONS)
  
  # for rel in curr_sro:
  #   if 'kissing' in rel:
  #     print(curr_sro)
  #     display(Image.fromarray(frame_rgb))
  
  object_list_of_str.append(", ".join(curr_sro))
  
  # sro_list.update(set(curr_sro))
  # print(curr_sro)
  if segment_index > 4:
    break

print(f"⏰ Ran scene graph {len(object_list_of_str), } frames in {(time.monotonic()-start_time):2f} seconds")  
