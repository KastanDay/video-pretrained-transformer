import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first

import sys
import json
import numpy as np
import pathlib
import jsonlines
import json
import time
import deeplake as dl
import sys
import ray
import more_itertools
import traceback
import pprint
import inspect
from ray.util.queue import Queue
from termcolor import colored

from deeplake_driver import DeeplakeManager
from clip_encoder import ClipEncoder

global NUM_PARALLEL_PROCESSES
global NUM_CPU_CORES
global NUM_GPUS
global GPU_PER_PROCESS

# datasets
BATCH_NAME              = 'parallel_15'
INPUT_DATASET_PATH      = f'/mnt/storage_ssd/v2_text_encode_results_{BATCH_NAME}'
RESULTS_DATASET_PATH    = f'/mnt/storage_ssd/clip_DUMMY_results_{BATCH_NAME}'

# THIS is GREAT balance on delta GPU, 4X GPU with clip running
# NUM_PARALLEL_PROCESSES = 20      # Number of parallel processes (limited by DRAM and SRAM)
# NUM_CPU_CORES = 64    # Numer of available physical cores to use (use max!)
# NUM_GPUS = 4          # Number of physical GPUs to use (use max)
# GPU_PER_PROCESS = 1/5 # threads per GPU, limited by OOM errors while also maximizing spread. 

NUM_PARALLEL_PROCESSES = 3      # Number of parallel processes (limited by DRAM and SRAM)
NUM_CPU_CORES = 12    # Numer of available physical cores to use (use max!)
NUM_GPUS = 1          # Number of physical GPUs to use (use max)
GPU_PER_PROCESS = 1 # threads per GPU, limited by OOM errors while also maximizing spread. 
BATCH_SIZE = 22 # 30 * 2 threads. good on 11GB

# rough GPU-mem per image is 22*3 / 11 = 6 images per gig.

@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES}, num_cpus=NUM_CPU_CORES/NUM_PARALLEL_PROCESSES, num_gpus=NUM_GPUS/NUM_PARALLEL_PROCESSES) 
class ParallelEncode:
  def __init__(self, all_clip_batches=None):
    
    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.db_manager = DeeplakeManager.remote(preprocessor_type='clip', database_path=RESULTS_DATASET_PATH, upload_queue=self.upload_queue)
    
    self.clip_batches_to_do = Queue()
    for batch in all_clip_batches:
      self.clip_batches_to_do.put(batch)
    
  @ray.method(concurrency_group="parallel_whisper_instances")  # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
  def parallel_clip_encode(self):
    '''
    Main function for parallel clip. 
    '''
    start = time.monotonic()
    process = ClipEncoder(debug=False)
    while self.clip_batches_to_do.qsize() > 0:
      batch = self.clip_batches_to_do.get(block=True)
      try:
        results_dict = process.run_clip_one_batch(batch)
        self.upload_queue.put(results_dict) # nearly instant, v fast 🏎💨
      except Exception as e:
        print(f"❌❌ Error during {inspect.currentframe().f_code.co_name}: {e}")
        traceback.print_exc()

      print(f"⏰ OVERALL TIME: clip-encoded {BATCH_SIZE} segments in {(time.monotonic() - start)/60:.2f} minutes. (time/frame = {((time.monotonic() - start)/BATCH_SIZE):.2f} sec)")
      start = time.monotonic()
    print(f"Worker done in {inspect.currentframe().f_code.co_name} (work queue empty), exiting! 😎")
      
def main():
  """ MAIN """
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
  print_cluster_stats()
  
  if False and os.path.exists(RESULTS_DATASET_PATH):
    print(f"Loading existing dataset from {RESULTS_DATASET_PATH}")
    ds = dl.load(RESULTS_DATASET_PATH) 
    segment_batch_list = add_samples_to_dict(ds, do_filtering=True)
  else:
    # create dataset  
    print(f"Creating new dataset at {RESULTS_DATASET_PATH}")
    ds = dl.deepcopy(INPUT_DATASET_PATH, RESULTS_DATASET_PATH, overwrite=True) 
    # CLIP produces FP32 embeddings.
    # inspo --> if 'clip_pooled_embedding' not in ds.tensors.keys():
    ds.create_tensor('clip_pooled_embedding',   htype='image', dtype=np.float32, sample_compression=None)
    ds.create_tensor('clip_last_hidden_states', htype='image', dtype=np.float32, sample_compression=None)
    ds.create_tensor('frames',                  htype='image', dtype=np.uint8,   sample_compression=None)
    print(ds.summary(), flush=True)
    print("Filling clip properties with all `None` so we can index and populate it")
    with ds:
      for _ in range(ds.max_len):
        ds.clip_pooled_embedding.append(None)
        ds.clip_last_hidden_states.append(None)
        ds.frames.append(None)
    ds.flush()
    print(ds.summary(), flush=True)
    # create batches
    segment_batch_list = add_samples_to_dict(ds, do_filtering=False)
  
  print("Num batches: ", len(segment_batch_list))
  print("Starting parallel batches")
  parallel_encode = ParallelEncode.remote(all_clip_batches=segment_batch_list)
  # only launch set number of workers, they all pull from the same work queue.
  all_done = ray.get([parallel_encode.parallel_clip_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)])
  print("Len of all threads: ", len(all_done))
  print("👉 Completed, finished main().")

def add_samples_to_dict(ds, do_filtering):
  '''
  make groups of standard size for Clip. Each dictionary contains 100 segments.
  I did it this way because videos are of variable size, but we want constant
  size to max GPU-memory usage.
  
  batch = {
    '<full_video_filepath>': [], # list of timestamp midpoints (1 per frame)
    '<full_video_filepath_2>': [], 
    ...
  }
  '''
  print("Starting filtering")
  start_time = time.monotonic()
  
  def add_one_sample(sample, batch, list_of_batches, total_samples):
    seg_start_time = float(sample['segment_metadata'].data()['value']['start'])
    seg_end_time   = float(sample['segment_metadata'].data()['value']['end'])
    midpoint = (seg_end_time + seg_start_time) / 2
    
    # add to dict {'<filepath>': [<frame_timestamp>, ...]}
    video_filepath = sample['video_filepath'].data()['value']
    if video_filepath not in batch.keys():
      batch[video_filepath] = []
    batch[video_filepath].append( {'timestamp': midpoint, 'db_index': idx} )
    
    # make batches of samples
    total_samples_in_batch = 0
    for video_filepath, time_and_db_index_list in batch.items():
      total_samples_in_batch += len(time_and_db_index_list)
    if total_samples_in_batch == BATCH_SIZE:
      list_of_batches.append(batch)
      batch = {}
    total_samples += 1
    return sample, batch, list_of_batches, total_samples
  
  # DO FILTERING HERE:
  batch = {}
  list_of_batches = []
  total_samples = 0
  for idx, sample in enumerate(ds):
    # filter completed CLIP results.
    if do_filtering:
      # only keep samples that have no results (because shape is 0)
      if sample.clip_pooled_embedding.numpy().shape == (0,) or sample.clip_last_hidden_states.numpy().shape == (0,):
        sample, batch, list_of_batches, total_samples = add_one_sample(sample, batch, list_of_batches, total_samples)
    else:
      # just add everything
      sample, batch, list_of_batches, total_samples = add_one_sample(sample, batch, list_of_batches, total_samples)
    
  # catch last batch, when smaller than 100.
  if batch != {}:
    list_of_batches.append(batch)
    batch = {}
  
  print(f"⏰ Time to filter completed results: {(time.monotonic() - start_time):.2f} seconds")
  assert len(list_of_batches) != 0 and batch == {}, "Error: list_of_batches is empty. nothing to process..."
  print(colored(f"✅ Already processed {ds.max_len - total_samples}", "green", attrs=["reverse", "bold"]))
  print(colored(f"👉 Total CLIP segments to process {total_samples}", "cyan", attrs=["reverse", "bold"]))
  return list_of_batches

def print_cluster_stats():
    print("Querying size of Ray cluster...\n")

    # print at start of staging
    print(f'''This cluster consists of
        {len(ray.nodes())} nodes in total
        {ray.cluster_resources()['CPU']} CPU cores in total
        {ray.cluster_resources()['memory']/1e9:.2f} GB CPU memory in total''')
    if ('GPU' in str(ray.cluster_resources())):
        print(f"        {ray.cluster_resources()['GPU']} GRAPHICCSSZZ cards in total")

if __name__ == '__main__':
    main()