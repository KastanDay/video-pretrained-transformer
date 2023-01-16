import os  # must be first

os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'  # must be first

import inspect
import json
import pathlib
import pprint
import sys
import time
import traceback
import math 

import deeplake as dl
import jsonlines
import more_itertools
import numpy as np
import ray
from clip_encoder import ClipEncoder
from deeplake_driver import DeeplakeManager
from ray.util.queue import Queue
from termcolor import colored
from tqdm import tqdm

# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
os.environ["RAY_memory_monitor_refresh_ms"] = "0" # prevents ray from killing the process when it runs out of memory

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray

global NUM_PARALLEL_PROCESSES
global NUM_CPU_CORES
global NUM_GPUS
global GPU_PER_PROCESS


# datasets
# BATCH_NAME = 'parallel_15'
# INPUT_DATASET_PATH = f'/mnt/storage_hdd/thesis/yt_1b_dataset/backups/v3_text_encode_results_{BATCH_NAME}'
# f'/mnt/storage_hdd/thesis/yt_1b_dataset/backups/v3_text_encode_results_{BATCH_NAME}'
# INPUT_DATASET_PATH = f'/tmp/v3_text_encode_results_{BATCH_NAME}'
# RESULTS_DATASET_PATH = f'/mnt/storage_hdd/thesis/yt_1b_dataset/backups/v3_CLIP_encode_results_{BATCH_NAME}' # hdd
# RESULTS_DATASET_PATH = f'/mnt/storage_ssd/v4_CLIP_encode_results_{BATCH_NAME}'  # ssd

BATCH_NAME = "handpicked_downloads"
INPUT_DATASET_PATH = f'/mnt/storage_hdd/thesis/handpicked_downloads/PREPROCESSED_DATA/text_encode_results_{BATCH_NAME}'
RESULTS_DATASET_PATH = f'/mnt/storage_hdd/thesis/handpicked_downloads/PREPROCESSED_DATA/CLIP_encode_results_{BATCH_NAME}'

# THIS is GREAT balance on delta GPU, 4X GPU with clip running
# NUM_PARALLEL_PROCESSES = 20      # Number of parallel processes (limited by DRAM and SRAM)
# NUM_CPU_CORES = 64    # Numer of available physical cores to use (use max!)
# NUM_GPUS = 4          # Number of physical GPUs to use (use max)
# GPU_PER_PROCESS = 1/5 # threads per GPU, limited by OOM errors while also maximizing spread.

NUM_PARALLEL_PROCESSES = 1  # Number of parallel processes (limited by DRAM and SRAM)
NUM_CPU_CORES = 12  # Numer of available physical cores to use (use max!)
NUM_GPUS = 1  # Number of physical GPUs to use (use max)
GPU_PER_PROCESS = 1  # threads per GPU, limited by OOM errors while also maximizing spread.
BATCH_SIZE = 30  # 30 * 2 threads. good on 11GB

# rough GPU-mem per image is 22*3 / 11 = 6 images per gig.


@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES},
            num_cpus=math.ceil(NUM_CPU_CORES / NUM_PARALLEL_PROCESSES),
            num_gpus=NUM_GPUS / NUM_PARALLEL_PROCESSES)
class ParallelEncode:
  '''
  Parallel actor. Degree of Parallelism = NUM_PARALLEL_PROCESSES
  __init__() is called only once. 
  parallel_text_encode() is called NUM_PARALLEL_PROCESSES times (and no more).
  '''

  def __init__(self, work_to_do_list=None):

    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.db_manager = DeeplakeManager.remote(preprocessor_type='clip',
                                             database_path=RESULTS_DATASET_PATH,
                                             upload_queue=self.upload_queue)

    self.batches_to_do_queue = Queue()
    for batch in work_to_do_list:
      self.batches_to_do_queue.put(batch)

  @ray.method(concurrency_group="parallel_whisper_instances")
  def parallel_clip_encode(self):
    '''
    Main function for parallel clip. 
    '''
    start = time.monotonic()
    process = ClipEncoder(debug=False)
    while self.batches_to_do_queue.qsize() > 0:
      print(f"📌 {self.batches_to_do_queue.qsize()} batches remaining")
      batch = self.batches_to_do_queue.get(block=True)
      try:
        results_dict = process.run_clip_one_batch(batch)
        self.upload_queue.put(results_dict)  # nearly instant, v fast 🏎💨
      except Exception as e:
        print(f"❌❌ Error during {inspect.currentframe().f_code.co_name}: {e}")
        traceback.print_exc()

      print(
          f"⏰ OVERALL TIME: clip-encoded {BATCH_SIZE} segments in {(time.monotonic() - start)/60:.2f} minutes. (time/frame = {((time.monotonic() - start)/BATCH_SIZE):.2f} sec)"
      )
      start = time.monotonic()
    print(f"Worker done in {inspect.currentframe().f_code.co_name} (work queue empty), exiting! 😎")


def main():
  """ MAIN """
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False,
           ignore_reinit_error=True)  # , num_gpus = 1
  print_cluster_stats()
  # print(colored(f"👉 Warning always creating new dataset", "yellow", attrs=["reverse", "bold"]))
  if os.path.exists(RESULTS_DATASET_PATH):
    print(f"Loading existing dataset from {RESULTS_DATASET_PATH}")
    ds = dl.load(RESULTS_DATASET_PATH)
    segment_batch_list = add_samples_to_dict(ds, do_filtering=True)
  else:
    # create dataset
    print(colored(f"⚠️ Starting from scratch, sleeping 3 sec to cancel..", "red", attrs=["reverse", "bold"]))
    time.sleep(3)
    print(f"Creating new dataset at {RESULTS_DATASET_PATH}")
    ds = dl.deepcopy(INPUT_DATASET_PATH, RESULTS_DATASET_PATH)  #, overwrite=True)
    # ds = dl.load(RESULTS_DATASET_PATH)
    # CLIP produces FP32 embeddings.
    # inspo --> if 'clip_pooled_embedding' not in ds.tensors.keys():
    with ds:
      ds.create_tensor('clip_pooled_embedding', htype='generic', dtype=np.float32, sample_compression='lz4')
      ds.create_tensor('clip_last_hidden_states', htype='generic', dtype=np.float32, sample_compression='lz4')
      ds.create_tensor('frames', htype='image', dtype=np.uint8, sample_compression='jpeg')
      ds.create_tensor('timestamp', htype='generic', dtype=float, sample_compression='lz4')
      # populate with none, so we can send it to the parallel workers.

      print("Filling clip properties with all `None` so we can index and populate it")
      start_time = time.monotonic()
      all_nones = [None] * ds.max_len  # previous method took 75 seconds
      ds.clip_pooled_embedding.extend(all_nones)
      ds.clip_last_hidden_states.extend(all_nones)
      ds.frames.extend(all_nones)
      ds.timestamp.extend(all_nones)
      print(ds.summary(), flush=True)
      print(f"⏰ Time to fill with Nones: {(time.monotonic() - start_time):.2f} seconds")

      start_time = time.monotonic()
      populate_ds_with_zeros().eval(ds, scheduler="ray", num_workers=NUM_CPU_CORES, skip_ok=True)
      print(f"⏰ Parallel.eval() Time to populate empty np arrays: {((time.monotonic() - start_time)/60):.2f} minutes")

      # # todo: could make this parallel like in text-encoder.
      # for _ in tqdm(range(ds.max_len),
      #               desc='populating',
      #               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
      #   ds.append(
      #       {
      #           'clip_pooled_embedding': np.zeros(1024, dtype=np.float32),
      #           'clip_last_hidden_states': np.zeros((577, 1024), dtype=np.float32),
      #           'frames': np.zeros((360, 640, 3), dtype=np.uint8),
      #           'timestamp': float(0),
      #       },
      #       skip_ok=True)  # skip tensors not in this dict (like the existing ones)
    ds.flush()
    print(ds.summary(), flush=True)
    # create batches
    start_time = time.monotonic()
    segment_batch_list = add_samples_to_dict(ds, do_filtering=False)
    print(f"⏰ add_samples_to_dict Filtering. Time to filter: {((time.monotonic() - start_time)/60):.2f} minutes")

  # segment_batch_list = segment_batch_list[:50] # for testing
  print("Num batches: ", len(segment_batch_list))
  print("Starting parallel batches")
  parallel_encode = ParallelEncode.remote(work_to_do_list=segment_batch_list)
  # only launch set number of workers, they all pull from the same work queue.
  all_done = ray.get([parallel_encode.parallel_clip_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)])
  print("Len of all threads: ", len(all_done))
  print("👉 Completed compute.")
  return
        
@dl.compute
def populate_ds_with_zeros(sample_in, sample_out):
  '''
  Pre-populate the dataset with zeros of proper shape. This makes it 100x faster to update later via indexing. 
  '''
  sample_out.clip_pooled_embedding.append(np.zeros(1024, dtype=np.float32))
  sample_out.clip_last_hidden_states.append(np.zeros((577, 1024), dtype=np.float32))
  sample_out.frames.append(np.zeros((360, 640, 3), dtype=np.uint8))
  sample_out.timestamp.append(float(0))
  return sample_out


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
    metadata = json.loads(sample['segment_metadata'].data()['value'])
    seg_start_time = float(metadata['start'])
    seg_end_time = float(metadata['end'])
    midpoint = (seg_end_time + seg_start_time) / 2

    # add to dict {'<filepath>': [<frame_timestamp>, ...]}
    video_filepath = sample['video_filepath'].data()['value']
    if video_filepath not in batch.keys():
      batch[video_filepath] = []
    batch[video_filepath].append({'timestamp': midpoint, 'db_index': idx})

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
    # filter already completed CLIP results.
    if do_filtering:
      # Test if numpy array contains only zeros (they're initialized that way)
      if not sample.clip_pooled_embedding.numpy().any():
        sample, batch, list_of_batches, total_samples = add_one_sample(sample, batch, list_of_batches, total_samples)
    else:
      # just add everything
      sample, batch, list_of_batches, total_samples = add_one_sample(sample, batch, list_of_batches, total_samples)

  # catch last batch, when smaller than BATCH_SIZE.
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

def await_ray_task_completion():
  print("Ensuring uploader is done before exiting.")
  # waiting for uploader (and any other jobs) to finish.
  while (ray.cluster_resources()['CPU'] != ray.available_resources()['CPU']):
    print(f"Uploader still in progress, some CPU cores still in use: {ray.available_resources()['CPU']} of {ray.cluster_resources()['CPU']}")
    time.sleep(5)

if __name__ == '__main__':
  main()
  await_ray_task_completion()
