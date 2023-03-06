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

# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # prevents ray from killing the process when it runs out of memory

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray

# for ray OOM errors: export RAY_DISABLE_MEMORY_MONITOR=1
GLOBAL_TOKENIZER = T5Tokenizer.from_pretrained("google/flan-t5-large")

# BATCH_NAME = "TVQA_BBT"
# INPUT_DATASET_PATH = f"/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/whisper_results_bbt_audios"
# RESULTS_DATASET_PATH = f"/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/feb_23_text_encode_results_{BATCH_NAME}"

BATCH_NAME = 'yt1b-val'
INPUT_DATASET_PATH = f'/mnt/teton/vpt/data/yt-1b_deeplake/feb_25_whisper_results_{BATCH_NAME}'
RESULTS_DATASET_PATH = f'/mnt/teton/vpt/data/yt-1b_deeplake/feb_25_text_encode_results_{BATCH_NAME}'

NUM_GPUS = 1
NUM_PARALLEL_PROCESSES = 16  # 16 works on 4090, but util is average 5%.
NUM_CPU_CORES = psutil.cpu_count()
BATCH_SIZE = 512

# batch_size 38 was max on 1080ti.


# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES}, num_cpus=0, num_gpus=NUM_GPUS)
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
    self.db_manager = DeeplakeManager.remote(preprocessor_type="text-encode",
                                             database_path=RESULTS_DATASET_PATH,
                                             upload_queue=self.upload_queue)
    return

  @ray.method(num_returns=1)
  def populate_work_queue(self,):
    print('starting queue')
    ds = dl.load(RESULTS_DATASET_PATH, read_only=True)  # read-only enables multiple to be open at once.
    print('ds loaded in queue')
    max_len = ds.max_len
    for i, sample in enumerate(ds):
      if not sample.done_text_encode.data()['value']:
        self.work_queue.put({'caption': sample.caption.text(), 'db_index': i})
      if i % 100 == 0:
        print(f"📌 {self.work_queue.qsize()} batches. Still adding more...")
    print("DONE POPULATING WORK QUEUE!")

    # block until work is done.
    while self.upload_queue.qsize() > 0 or self.work_queue.qsize() > 0:
      time.sleep(10)

    print("✅ Work & upload queue are empty. All work should be done! ")
    return 0

  @ray.method(concurrency_group="parallel_whisper_instances")
  def parallel_text_encode(self):
    """
    Main function for parallel whisper.
    """
    process = FlanT5Encoder(device="cuda:0")
    while self.work_queue.qsize() > 0:
      start = time.monotonic()
      print(f"📌 {self.work_queue.qsize()} batches remaining")
      try:
        batch = self.work_queue.get(block=True, timeout=10)
      except Exception as e:
        # it'll raise Empty after timeout, so just test while loop condition
        print("Temout waiting for work from work_queue. This is expected near end of job as workers finish.")
        continue

      try:
        # returns: list of np.arrays, each of different shape [NUM_TOKENS, 1024]
        last_hidden_states_batch = process.encode(batch)
        caption_embed_dict_list = []
        for embed in last_hidden_states_batch:
          caption_embed_dict_list.append({"db_index": embed["db_index"], "last_hidden_states": embed["last_hidden_states"]})
        ## ADD TO DATASET (via upload queue)
        self.upload_queue.put(caption_embed_dict_list)
        # print("Added to Queue!")
      except Exception as e:
        print("❌❌Error during text-encode: ", e)
        traceback.print_exc()
        # pprint.pprint(caption_embed_dict_list)
      print(f"⏰ Time to Text-encode file: {(time.monotonic() - start)/60:.2f} minutes."
            "(time/segment): {((time.monotonic() - start)/BATCH_SIZE):.2f} sec")


@dl.compute
def populate_ds_with_zeros(sample_in, sample_out):
  # assert type(sample_in_caption) == str or type(sample_in_caption) == np.str_, print(f"expecting just the pure caption. got {type(sample_in_caption)}")
  caption = sample_in.caption.data()["value"]
  tokenized = GLOBAL_TOKENIZER(caption, return_tensors="pt", truncation=False).input_ids
  # sample_out.caption_embedding.append(np.negative(np.ones((len(tokenized[0]), 1024)), dtype=np.float16))
  sample_out.caption_embedding.append(np.zeros((len(tokenized[0]), 1024), dtype=np.float16))
  return sample_out


# def filter_completed_text_encodes(ds):
#   index_caption_pairs = []
#   print("Filtering already completed text-encodes...")
#   start_time = time.monotonic()
#   for idx, sample in tqdm(
#       enumerate(ds),
#       desc="filtering completed",
#       total=ds.max_len,
#       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
#   ):
#     try:
#       # if not done yet, add to processing queue index_caption_pairs
#       # print("WARNING THIS WILL NOT WORK BECAUSE THE EMPTY TENSORS ARE ALL -1s NOW. otherwise they're empty.")
#       if not sample.caption_embedding.numpy().any():
#         index_caption_pairs.append({"db_index": idx, "caption": sample.caption.data()["value"]})
#     except IndexError as e:
#       print(e)
#       traceback.print_exc()
#       # if there's an IndexError, then the caption_embedding is empty. Caused by bug in compression code.
#       index_caption_pairs.append({"db_index": idx, "caption": sample.caption.data()["value"]})
#   print(f"⏰ Time to filter: {((time.monotonic() - start_time)/60):.2f} minutes")
#   return index_caption_pairs

# def parallel_filter_completed_text_encodes(sample_in, sample_out):
#   index_caption_pairs = []
#   try:
#     # if not done yet, add to processing queue index_caption_pairs
#     if not sample_in.caption_embedding.numpy().any():
#       index_caption_pairs.append({"db_index": idx, "caption": sample.caption.data()["value"]})
#   except IndexError as e:
#     print(e)
#     traceback.print_exc()
#     # if there's an IndexError, then the caption_embedding is empty. Caused by bug in compression code.
#     index_caption_pairs.append({"db_index": idx, "caption": sample.caption.data()["value"]})
#   print(f"⏰ Time to filter: {((time.monotonic() - start_time)/60):.2f} minutes")
#   return index_caption_pairs


def main():
  """MAIN"""
  index_caption_pairs = []  # list of: {'db_index': int, 'caption': str}
  # todo: check for completed segments (that already have a caption_embedding)
  if os.path.exists(RESULTS_DATASET_PATH):
    pass
    # ds = dl.load(RESULTS_DATASET_PATH)
    # print(ds.summary())
    # index_caption_pairs = filter_completed_text_encodes(ds)

  else:
    # Create output database (none exists yet)
    print(colored(f"👉 Creating output database at {RESULTS_DATASET_PATH}", "cyan", attrs=["reverse", "bold"]))
    output_ds = dl.deepcopy(INPUT_DATASET_PATH, RESULTS_DATASET_PATH, overwrite=True)
    with output_ds:
      # tf_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type() # couldn't get this working weird imports.
      output_ds.create_tensor("caption_embedding", htype="generic", dtype=np.float16, sample_compression=None)
      output_ds.create_tensor("done_text_encode", htype="generic", dtype=bool, sample_compression=None)
      output_ds.caption_embedding.extend([np.float16(0)] * output_ds.max_len)  # make equal size (fastest way)
      output_ds.done_text_encode.extend([False] * output_ds.max_len)  # set all to false (fastest way)
      output_ds.flush()
    with output_ds:
      print("Prepopulating `caption_embedding` tensor with custom-tokenized np.zeros((custom_token_len, 1024).")
      print(output_ds.summary())
      populate_ds_with_zeros().eval(output_ds, scheduler="ray", num_workers=NUM_CPU_CORES, skip_ok=True)
      print("Output ds after prepopulating")
      print(output_ds.summary())

    del output_ds  # hopefully this closes connection?
    # print(colored(f"TODO: I need to write code to populate index_caption_pairs", "red", attrs=["reverse", "bold"]))

    # parallel_filter_completed_text_encodes().eval(output_ds,
    #                                               scheduler="ray",
    #                                               num_workers=NUM_PARALLEL_PROCESSES,
    #                                               skip_ok=True)
    # index_caption_pairs = filter_completed_text_encodes(output_ds)

  # if len(index_caption_pairs) == 0:
  #   print(colored(f"No new captions to encode. Exiting!", "green", attrs=["reverse", "bold"]))
  #   exit()
  # else:
  #   print(
  #       colored(
  #           f"👉 Starting to encode these text-captions: {len(index_caption_pairs)}",
  #           "cyan",
  #           attrs=["reverse", "bold"],
  #       ))

  # create batches of length BATCH_SIZE --- SHOULD ADD THIS BACK IN.
  # if BATCH_SIZE == 1:
  #   batches = [index_caption_pairs]
  # else:
  #   batches = list(more_itertools.chunked(index_caption_pairs, BATCH_SIZE))
  # print("Num batches: ", len(batches))

  # print(
  #     colored(
  #         f"TODO: 👉 Ensure that I'm only keeping hidden states that are non-padding tokens",
  #         "yellow",
  #         attrs=["reverse", "bold"],
  #     ))

  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False, ignore_reinit_error=True)
  print_cluster_stats()

  # only launch set number of workers, they all pull from the same work queue.
  parallel_encode = ParallelEncode.remote()
  print("Starting upload queue")
  populate_work_queue_future = parallel_encode.populate_work_queue.remote()
  print("Starting parallel batches")
  all_done_futures = [parallel_encode.parallel_text_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)]
  all_done = ray.get(all_done_futures)
  all_done.append(ray.get(populate_work_queue_future))

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


# def write_error(file):
#   failed_file_json_object = json.dumps(str(file))
#   empty_filepath = LOCAL_VIDEO_DIR + "_whisper_empty.jsonl"
#   if not os.path.exists(empty_filepath):
#     pathlib.Path(empty_filepath).touch()
#   with jsonlines.open(empty_filepath, mode="a") as writer:
#     writer.write({"video_filepath": failed_file_json_object})


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
