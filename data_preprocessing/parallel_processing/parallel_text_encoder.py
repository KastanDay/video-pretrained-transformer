print("Use env: eval_clip_t5_torch19")
import os 
import json
import numpy as np
from PIL import Image
import pathlib
from pathlib import Path
import glob
import jsonlines
import json
import time
import deeplake as dl
import tqdm
import sys
import ray
import traceback
import more_itertools
import pprint
from ray.util.queue import Queue
from termcolor import colored

# for ray OOM errors: export RAY_DISABLE_MEMORY_MONITOR=1

from deeplake_driver import DeeplakeManager
from text_encoder import FlanT5Encoder

BATCH_NAME                    = 'parallel_15'
WHISPER_RESULTS_DATASET_PATH  = f'/mnt/storage_ssd/v2_text_encode_results_{BATCH_NAME}'
INPUT_DATASET_PATH            = f'/mnt/storage_ssd/no_compression_whisper_results_{BATCH_NAME}'

NUM_GPUS = 1
NUM_PARALLEL_PROCESSES = 14
NUM_CPU_CORES = 12

@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES}, num_cpus=NUM_CPU_CORES, num_gpus=NUM_GPUS) 
class ParallelTextEncode:
  def __init__(self):
    
    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.db_manager = DeeplakeManager.remote(database_path=WHISPER_RESULTS_DATASET_PATH, upload_queue=self.upload_queue)
    
  @ray.method(concurrency_group="parallel_whisper_instances")  # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
  def parallel_caption_extraction(self, text_batch):
    '''
    Main function for parallel whisper. 
    '''
    start = time.monotonic()
    process = FlanT5Encoder()
    for index_caption_dict in text_batch:
      try:
        # MAIN: run text-encode
        last_hidden_states = process.encode(index_caption_dict['caption'])
        last_hidden_states = last_hidden_states.detach().cpu().numpy()
        last_hidden_states = last_hidden_states.reshape(-1, 1024)
        index_caption_dict = {'db_index': index_caption_dict['db_index'], 'caption': index_caption_dict['caption'], 'last_hidden_states': last_hidden_states}
        ## ADD TO DATASET (via upload queue)
        print(f"🔥 itr: {index_caption_dict['db_index']}. About to add work to queue")
        self.upload_queue.put(index_caption_dict)
        print("Added to Queue!")
      except Exception as e:
        print("❌❌Error during text-encode: ", e)
        print(index_caption_dict)
        traceback.print_exc()
        # write failed files to jsonlines
        write_error(index_caption_dict)

      # one file done        
      print(f"⏰ Time to Text-encode file: {(time.monotonic() - start)/60:.2f} minutes")
      start = time.monotonic()

def main():
  """ MAIN """
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False, ignore_reinit_error=True) # , num_gpus = 1
  print_cluster_stats()

  index_caption_pairs = [] #list of: {'db_index': int, 'caption': str}
  # todo: check for completed segments (that already have a caption_embedding)
  if os.path.exists(WHISPER_RESULTS_DATASET_PATH):
    ds = dl.load(WHISPER_RESULTS_DATASET_PATH)
    print(ds.summary())
    for idx, sample in enumerate(ds):
      try:
        # if not done yet, add to processing queue index_caption_pairs 
        if sample.caption_embedding.numpy().shape == (0, 0):
          index_caption_pairs.append( {'db_index': idx, 'caption': sample.caption.data()['value']} )
      except IndexError as e:
        print(e)
        # if there's an IndexError, then the caption_embedding is empty. Caused by bug in compression code.
        index_caption_pairs.append( {'db_index': idx, 'caption': sample.caption.data()['value']} )
  else:
    # Create output database (none exists yet)
    print("Creating output database at ", WHISPER_RESULTS_DATASET_PATH)
    output_ds = dl.deepcopy(INPUT_DATASET_PATH, WHISPER_RESULTS_DATASET_PATH, overwrite=True) 
    with output_ds:
      # all compression is FUCKED
      output_ds.create_tensor('caption_embedding', htype='image', dtype=np.float16, sample_compression=None) 
      # output_ds.caption_embedding[0] = None
      for idx in range(output_ds.max_len):
        output_ds.caption_embedding.append(None)
        index_caption_pairs.append( {'db_index': idx, 'caption': output_ds.caption[idx].data()['value']} )
    print(output_ds.summary())
  
  if len(index_caption_pairs) == 0:
    print(colored(f"No new captions to encode. Exiting!", "green", attrs=["reverse", "bold"]))
    exit()
  else:
    print(colored(f"👉 Starting to encode these text-captions: {len(index_caption_pairs)}", "cyan", attrs=["reverse", "bold"]))
  
  # split files into batches
  if NUM_PARALLEL_PROCESSES == 1:
    batches = [index_caption_pairs]
  else:
    batches = list(more_itertools.divide(NUM_PARALLEL_PROCESSES, index_caption_pairs))
  print("Num batches: ", len(batches))
  assert len(batches) == (NUM_PARALLEL_PROCESSES), "there is supposed to be one Ray thread per batch"

  print("Starting parallel batches")
  parallel_text_encode = ParallelTextEncode.remote()
  all_done = ray.get([parallel_text_encode.parallel_caption_extraction.remote(batch) for batch in batches])
  print("Len of all threads: ", len(all_done))
  print("👉 Completed, finished main().")

def print_cluster_stats():
    print("Querying size of Ray cluster...\n")
    # print at start of staging
    print(f'''This cluster consists of
        {len(ray.nodes())} nodes in total
        {ray.cluster_resources()['CPU']} CPU cores in total
        {ray.cluster_resources()['memory']/1e9:.2f} GB CPU memory in total''')
    if ('GPU' in str(ray.cluster_resources())):
        print(f"        {ray.cluster_resources()['GPU']} GRAPHICCSSZZ cards in total")

def write_error(file):
  failed_file_json_object = json.dumps(str(file))
  empty_filepath = LOCAL_VIDEO_DIR + "_whisper_empty.jsonl"
  if not os.path.exists(empty_filepath):
      pathlib.Path(empty_filepath).touch()
  with jsonlines.open(empty_filepath, mode='a') as writer:
      writer.write({"video_filepath": failed_file_json_object}) 

if __name__ == '__main__':
    main()