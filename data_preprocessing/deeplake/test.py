import deeplake
dataset_name = '/mnt/storage_ssd/v13_parallel_ingest_p15'
read_lake = deeplake.load(dataset_name)

print(read_lake.summary())

# read_lake.segment_metadata[2].data()['value'][total_segments]
# read_lake.text_caption_embeddings[2].data()['value']

import cv2
from PIL import Image
for i in range(3):
  # display(Image.fromarray(read_lake.segment_frames[i].numpy()))
  
  frame = read_lake.segment_frames[i].numpy()
  print(type(frame))
  print(frame.shape)
  
  frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
  # cv2.imshow("image", read_lake.segment_frames[i].numpy())
  
  