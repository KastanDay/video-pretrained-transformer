print("Use conda env: vpt")
import glob
import json
import os
import pathlib
import pprint
import sys
import time
import traceback
from pathlib import Path

import deeplake as dl
import jsonlines
import more_itertools
import numpy as np
import psutil
import ray
from deeplake_driver import DeeplakeManager
from PIL import Image
from ray.util.queue import Queue
from termcolor import colored
from text_encoder import FlanT5Encoder
from tqdm import tqdm
from transformers import T5Tokenizer

os.environ['TRANSFORMERS_CACHE'] = '/mnt/teton/utils/cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/mnt/teton/utils/cache/datasets'

# our own code
from clip_encoder import ClipEncoder
from text_encoder import FlanT5Encoder
from TVQA_eval import TVQA_Eval

# sys.path.append("../../model/good_files")
# sys.path.append("../../model/good_files")
# sys.path.append("/home/k/video-pretrained-transformer/model/good_files")

# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # prevents ray from killing the process when it runs out of memory

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray

# for ray OOM errors: export RAY_DISABLE_MEMORY_MONITOR=1
GLOBAL_TOKENIZER = T5Tokenizer.from_pretrained("google/flan-t5-xxl")

# BATCH_NAME = "TVQA_BBT"
# INPUT_DATASET_PATH = f"/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/whisper_results_bbt_audios"
# RESULTS_DATASET_PATH = f"/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/feb_23_text_encode_results_{BATCH_NAME}"

BATCH_NAME = 'tvqa_whole'
RESULTS_DATASET_PATH = f'/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/mar_28_TVQA_encode_{BATCH_NAME}'

NUM_GPUS = 2
NUM_PARALLEL_PROCESSES = 1  # 16 works on 4090, but util is average 5%.
NUM_CPU_CORES = psutil.cpu_count()
BATCH_SIZE = 512

# batch_size 38 was max on 1080ti.


# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES}, num_cpus=0, num_gpus=2)  # 2 gpu for CLIP + text
class ParallelEncode:
  """
  Parallel actor. Degree of Parallelism = NUM_PARALLEL_PROCESSES
  __init__() is called only once.
  parallel_text_encode() is called NUM_PARALLEL_PROCESSES times (and no more).
  """

  def __init__(self, work_to_do_list=None):
    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.work_queue = Queue()
    self.db_manager = DeeplakeManager.remote(preprocessor_type="tvqa-encode",
                                             database_path=RESULTS_DATASET_PATH,
                                             upload_queue=self.upload_queue)
    return

  @ray.method(num_returns=1)
  def populate_work_queue(self, train_filepath):
    '''
    Work queue looks like 1 row from train_qa_json.
    Example:
    {'a0': 'Cafeteria', 'a1': 'Hallway', 'a2': 'Car', 'a3': 'Patients room', 'a4': 'Outside', 'answer_idx': 4, 'q': 'Where is Meredith when George approaches her?', 'qid': 0, 'show_name': "Grey's Anatomy", 'ts': '76.01-84.2', 'vid_name': 'grey_s03e20_seg02_clip_14'}
    '''

    print(f'loading train_qa_json from: {train_filepath}')
    with open(train_filepath, 'r') as f:
      self.train_qa_json = [json.loads(line) for line in f]

    print('Populating work queue')
    for i, question in enumerate(self.train_qa_json):
      self.work_queue.put(question)
      if i - 1 % 100 == 0:
        print(f"📌 {self.work_queue.qsize()} batches. Still adding more...")
    print("✅ DONE POPULATING WORK QUEUE!")

    # block until work is done.
    while self.upload_queue.qsize() > 0 or self.work_queue.qsize() > 0:
      time.sleep(10)

    print("✅ Work & upload queue are empty. All work should be done! ")
    return 0

  @ray.method(concurrency_group="parallel_whisper_instances")
  def parallel_tvqa_encode(self):
    """
    Main function for TVQA encoding.
    """
    tvqa_eval = TVQA_Eval()

    while self.work_queue.qsize() > 0:
      start = time.monotonic()
      print(f"📌 {self.work_queue.qsize()} batches remaining")
      try:
        train_sample = self.work_queue.get(block=True, timeout=10)
      except Exception as e:
        # it'll raise Empty after timeout, so just test while loop condition
        print("Temout waiting for work from work_queue. This is expected near end of job as workers finish.")
        continue

      try:
        # RUN MAIN MODELS
        if train_sample['show_name'] == 'The Big Bang Theory':
          continue
        context_vector_list = tvqa_eval.create_context_vectors(train_sample)
        ans_list = tvqa_eval.get_answers_from_question(train_sample)
        ## ADD TO DATASET (via upload queue)
        self.upload_queue.put((context_vector_list, ans_list))
      except FileNotFoundError as e:
        # this is EXPECTED as some videos are missing somehow.
        print(e)
        print(f"WARNING: Could not find video {train_sample['vid_name']}. Skipping...")
      except Exception as e:
        print("❌❌Error during parallel_TVQA_encode: ", e)
        traceback.print_exc()
        # pprint.pprint(caption_embed_dict_list)
      print(f"⏰ Time to Text-encode file: {(time.monotonic() - start)/60:.2f} minutes."
            "(time/segment): {((time.monotonic() - start)/BATCH_SIZE):.2f} sec")

  def get_upload_queue_size(self):
    '''
    These 'get queue size' are used in main() to ensure we finish all work before exiting.
    '''
    return self.upload_queue.qsize()

  def get_work_queue_size(self):
    '''
    These 'get queue size' are used in main() to ensure we finish all work before exiting.
    '''
    return self.work_queue.qsize()


@dl.compute
def populate_ds_with_zeros(sample_in, sample_out):
  # assert type(sample_in_caption) == str or type(sample_in_caption) == np.str_, print(f"expecting just the pure caption. got {type(sample_in_caption)}")
  caption = sample_in.caption.data()["value"]
  tokenized = GLOBAL_TOKENIZER(caption, return_tensors="pt", truncation=False).input_ids
  # sample_out.caption_embedding.append(np.negative(np.ones((len(tokenized[0]), 1024)), dtype=np.float32))
  sample_out.caption_embedding.append(np.zeros((len(tokenized[0]), 1024), dtype=np.float32))
  return sample_out


# iterate over the train. pass to create_context_vectors
# /mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_qa_release/tvqa_train.jsonl


def main():
  """MAIN"""
  # todo: check for completed segments (that already have a caption_embedding)
  if os.path.exists(RESULTS_DATASET_PATH):
    pass
    # ds = dl.load(RESULTS_DATASET_PATH)
    # print(ds.summary())
    # index_caption_pairs = filter_completed_text_encodes(ds)

  else:
    # Create output database (none exists yet)
    print(colored(f"👉 Creating output database at {RESULTS_DATASET_PATH}", "cyan", attrs=["reverse", "bold"]))
    output_ds = dl.empty(RESULTS_DATASET_PATH, overwrite=True)
    with output_ds:
      # tf_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type() # couldn't get this working weird imports.
      output_ds.create_tensor("context_vector", htype="generic", dtype=np.float32, sample_compression=None)
      output_ds.create_tensor("label", htype="text", dtype=str, sample_compression=None)
      # output_ds.create_tensor("done_text_encode", htype="generic", dtype=bool, sample_compression=None)

      # NO NEED to prepopulate. We'll just append instead. no need for ordering.
      # total_samples = 650_000  # train samples * num questions
      # output_ds.context_vector.extend([np.float32(0)] * total_samples)  # make equal size (fastest way)
      output_ds.flush()
    del output_ds  # hopefully this closes connection?

  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False, ignore_reinit_error=True)
  print_cluster_stats()

  train_filepath = "/mnt/teton/vpt/data/benchmark_datasets/TVQA/TVQA/data/tvqa_qa_release/tvqa_train.jsonl"

  # only launch set number of workers, they all pull from the same work queue.
  parallel_encode = ParallelEncode.remote()
  print("Starting upload queue")
  populate_work_queue_future = parallel_encode.populate_work_queue.remote(train_filepath)
  print("Starting parallel batches")
  all_done_futures = [parallel_encode.parallel_tvqa_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)]
  all_done = ray.get(all_done_futures)
  all_done.append(ray.get(populate_work_queue_future))

  ## THIS is the best way to ensure work is done before exiting.
  while ray.get(parallel_encode.get_upload_queue_size.remote()) > 0 or ray.get(parallel_encode.get_work_queue_size.remote()) > 0:
    print("Deeplake upload queue size", ray.get(parallel_encode.get_upload_queue_size.remote()))
    print("Text-encode work queue size", ray.get(parallel_encode.get_work_queue_size.remote()))
    print("Still uploading files, sleeping 5 seconds..")
    time.sleep(5)
  print("✅ All work and uploads should be done, exiting!")

  print("Len of all threads: ", len(all_done))
  print("👉 Completed, finished main().")
  return 0


def print_cluster_stats():
  print("Querying size of Ray cluster...\n")
  # print at start of staging
  print(f"""This cluster consists of
        {len(ray.nodes())} nodes in total
        {ray.cluster_resources()['CPU']} CPU cores in total
        {ray.cluster_resources()['memory']/1e9:.2f} GB CPU memory in total""")
  if "GPU" in str(ray.cluster_resources()):
    print(f"        {ray.cluster_resources()['GPU']} GRAPHICCSSZZ cards in total")


def await_ray_task_completion():
  '''
  Wait for uploader (and any other jobs) to finish.
  # optional improvement: check the `ray memory` for remining objects to be uploaded.
  '''
  print("Ensuring uploader is done before exiting.")
  while (ray.cluster_resources()['CPU'] != ray.available_resources()['CPU']):
    print(
        f"Uploader still in progress, some CPU cores still in use: {ray.available_resources()['CPU']} of {ray.cluster_resources()['CPU']}")
    time.sleep(5)


if __name__ == '__main__':
  main()
  await_ray_task_completion()
