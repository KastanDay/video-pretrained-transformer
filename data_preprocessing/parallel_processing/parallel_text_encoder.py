print("Use env: nlp_v2")
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
import sys
import ray
import traceback
import more_itertools
import pprint
from ray.util.queue import Queue
from termcolor import colored
from tqdm import tqdm

# for ray OOM errors: export RAY_DISABLE_MEMORY_MONITOR=1


from deeplake_driver import DeeplakeManager
from text_encoder import FlanT5Encoder

BATCH_NAME              = 'parallel_15'
RESULTS_DATASET_PATH    = f'/mnt/storage_ssd/v3_text_encode_results_{BATCH_NAME}'
INPUT_DATASET_PATH      = f'/mnt/storage_ssd/FULL_whisper_results_{BATCH_NAME}'

NUM_GPUS = 1
NUM_PARALLEL_PROCESSES = 11
NUM_CPU_CORES = 12
BATCH_SIZE = 30

@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES}, num_cpus=NUM_CPU_CORES, num_gpus=NUM_GPUS) 
class ParallelEncode:
  '''
  Parallel actor. Degree of Parallelism = NUM_PARALLEL_PROCESSES
  __init__() is called only once. 
  parallel_text_encode() is called NUM_PARALLEL_PROCESSES times (and no more).
  '''
  def __init__(self, work_to_do_list=None):
    
    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.db_manager = DeeplakeManager.remote(preprocessor_type='text-encode', database_path=RESULTS_DATASET_PATH, upload_queue=self.upload_queue)
    
    self.work_queue = Queue()
    self.populate_work_queue.remote(work_to_do_list) # non-blocking
  
  @ray.remote
  def populate_work_queue(self, work_to_do_list):
    for batch in work_to_do_list:
      self.work_queue.put(batch)
    
  @ray.method(concurrency_group="parallel_whisper_instances")  # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
  def parallel_text_encode(self):
    '''
    Main function for parallel whisper. 
    '''
    process = FlanT5Encoder()
    while self.work_queue.qsize() > 0:
      start = time.monotonic()
      batch = self.work_queue.get(block=True)
      try:
        # returns: list of np.arrays, each of different shape [NUM_TOKENS, 1024]
        last_hidden_states_batch = process.encode(batch)
        caption_embed_dict_list = []
        for input_batch_item, embed in zip(batch, last_hidden_states_batch):
           caption_embed_dict_list.append({
            'db_index': input_batch_item['db_index'], 
            'last_hidden_states': embed
          })
        ## ADD TO DATASET (via upload queue)
        # print(f"🔥 itr: {index_caption_dict['db_index']}. About to add work to queue")
        self.upload_queue.put(caption_embed_dict_list)
        # print("Added to Queue!")
      except Exception as e:
        print("❌❌Error during text-encode: ", e)
        traceback.print_exc()
        pprint.pprint(caption_embed_dict_list)
      print(f"⏰ Time to Text-encode file: {(time.monotonic() - start)/60:.2f} minutes. (time/segment): {((time.monotonic() - start)/BATCH_SIZE):.2f} sec")

from transformers import T5Tokenizer
GLOBAL_TOKENIZER = T5Tokenizer.from_pretrained("google/flan-t5-large")
'''
Fun idea... maybe not worth implementing.
'''
@dl.compute
def populate_tensor(sample_in, sample_out):
  # assert type(sample_in_caption) == str or type(sample_in_caption) == np.str_, print(f"expecting just the pure caption. got {type(sample_in_caption)}")
  caption = sample_in.caption.data()['value']
  tokenized = GLOBAL_TOKENIZER(caption, return_tensors="pt", truncation=False).input_ids
  # sample_out.caption_embedding.append(np.negative(np.ones((len(tokenized[0]), 1024)), dtype=np.float16))
  sample_out.caption_embedding.append(np.zeros((len(tokenized[0]), 1024), dtype=np.float16))
  return sample_out

def main():
  """ MAIN """

  index_caption_pairs = [] #list of: {'db_index': int, 'caption': str}
  # todo: check for completed segments (that already have a caption_embedding)
  if os.path.exists(RESULTS_DATASET_PATH):
    ds = dl.load(RESULTS_DATASET_PATH)
    print(ds.summary())
    start_time = time.monotonic()
    for idx, sample in enumerate(ds):
      try:
        # if not done yet, add to processing queue index_caption_pairs 
        # print("WARNING THIS WILL NOT WORK BECAUSE THE EMPTY TENSORS ARE ALL -1s NOW. otherwise they're empty.")
        print("Filtering already completed text-encodes...")
        if not sample.caption_embedding.numpy().any():
          index_caption_pairs.append( {'db_index': idx, 'caption': sample.caption.data()['value']} )
      except IndexError as e:
        print(e)
        traceback.print_exc()
        # if there's an IndexError, then the caption_embedding is empty. Caused by bug in compression code.
        index_caption_pairs.append( {'db_index': idx, 'caption': sample.caption.data()['value']} )
    print(f"⏰ Time to filter: {((time.monotonic() - start_time)/60):.2f} minutes")
  else:
    # Create output database (none exists yet)
    print(colored(f"👉 Creating output database at {RESULTS_DATASET_PATH}", "cyan", attrs=["reverse", "bold"]))
    output_ds = dl.deepcopy(INPUT_DATASET_PATH, RESULTS_DATASET_PATH, overwrite=True) 
    with output_ds:
      output_ds.create_tensor('caption_embedding', htype='generic', dtype=np.float16, sample_compression=None) 
      output_ds.caption_embedding.extend([np.float16(0)] * output_ds.max_len) # make equal size (fastest way)
      output_ds.flush()
    with output_ds:
      # don't use compression. all compression is FUCKED in Deeplake... 
      # all_captions_list = []
      # for idx in tqdm(range(output_ds.max_len), desc='grabbing captions', total=output_ds.max_len, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
        # todo: should I tokenize the captions here just to ensure we set the right size from the start? 
        #  \__ Might be good to avoid corruption errors with many resizes.
        # todo: make this parallel. it's so slow.
        # caption = output_ds.caption[idx].data()['value']
        # all_captions_list.append(caption)
        # index_caption_pairs.append( {'db_index': idx, 'caption': caption} )
        # output_ds.caption_embedding[idx] = None # np.float16(0)
      print("Prepopulating `caption_embedding` tensor with custom-tokenized np.zeros((custom_token_len, 1024).")
      # print(len(all_captions_list))
      # populate_tensor().eval(all_captions_list, output_ds, scheduler='ray', num_workers=11)
      # populate_tensor().eval(output_ds, scheduler='threaded', num_workers=11)
      populate_tensor().eval(output_ds, scheduler='ray', num_workers=11, skip_ok=True)
      print("Output ds after prepopulating")
      print(output_ds.summary())
  
  if len(index_caption_pairs) == 0:
    print(colored(f"No new captions to encode. Exiting!", "green", attrs=["reverse", "bold"]))
    exit()
  else:
    print(colored(f"👉 Starting to encode these text-captions: {len(index_caption_pairs)}", "cyan", attrs=["reverse", "bold"]))
  
  # create batches of length BATCH_SIZE
  if BATCH_SIZE == 1:
    batches = [index_caption_pairs]
  else:
    batches = list(more_itertools.chunked(index_caption_pairs, BATCH_SIZE))
  print("Num batches: ", len(batches))
  
  print(colored(f"TODO: 👉 Ensure that I'm only keeping hidden states that are non-padding tokens", "yellow", attrs=["reverse", "bold"]))

  print("Starting parallel batches")
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False, ignore_reinit_error=True) # , num_gpus = 1
  print_cluster_stats()
  
  parallel_encode = ParallelEncode.remote(work_to_do_list=batches)
  # only launch set number of workers, they all pull from the same work queue.
  all_done = ray.get([parallel_encode.parallel_text_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)])
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