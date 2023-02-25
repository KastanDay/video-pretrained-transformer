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
import tqdm
from deeplake_driver import DeeplakeManager
from PIL import Image
from ray.util.queue import Queue

sys.path.append("../whisper_audio")
from CaptionPreprocessing import CaptionPreprocessing

# TODO: Set max_restarts and max_task_retries to enable retry when the task crashes due to OOM.
os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # prevents ray from killing the process when it runs out of memory

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray

BATCH_NAME = 'bbt_audios'
INPUT_VIDEOS_PATH = f'/mnt/teton/vpt/data/benchmark_datasets/TVQA/uncompressed_audio/bbt_new/{BATCH_NAME}'
WHISPER_RESULTS_DATASET_PATH = f'/mnt/teton/vpt/data/benchmark_datasets/TVQA/_deeplake/whisper_results_{BATCH_NAME}'
LOCAL_VIDEO_DIR = f'/tmp/{BATCH_NAME}'  # used for wavs

NUM_GPUS = 1
NUM_PARALLEL_INSTANCES = 2
NUM_CPU_CORES = psutil.cpu_count()


@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_INSTANCES}, num_cpus=0, num_gpus=NUM_GPUS)
class ParallelWhisper:

  def __init__(self):

    # Every parallel_caption_extraction writes to this queue. Then the uploader pulls from it. Magic.
    self.upload_queue = Queue()
    self.db_manager = DeeplakeManager.remote(preprocessor_type='whisper',
                                             database_path=WHISPER_RESULTS_DATASET_PATH,
                                             upload_queue=self.upload_queue)

  @ray.method(concurrency_group="parallel_whisper_instances"
             )  # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
  def parallel_caption_extraction(self, file_batch):
    '''
    Main function for parallel whisper. 
    '''
    sys.path.append("../whisper_audio")
    from CaptionPreprocessing import CaptionPreprocessing

    # print("WARNING HARD CODING CUDA:1")
    process = CaptionPreprocessing(device='cuda:0')
    for file in file_batch:
      start = time.monotonic()
      try:
        # MAIN: run whisper
        process.load_mp4_to_wav_with_outpath(file, out_dir=LOCAL_VIDEO_DIR + "_wav")
        whisper_one_video_results = process.get_segments_thresholded()
        # If there are no captions/if it is non-english
        if not whisper_one_video_results:
          print("File is empty!")
          write_error(file)
        else:
          ## ADD TO DATASET (via upload queue)
          print("🔥 About to add work to queue")
          self.upload_queue.put(whisper_one_video_results)
          print("Added to Queue!")
      except Exception as e:
        print(file)
        # write failed files to jsonlines
        print(f"❌❌ Error during whisper: {e}")
        traceback.print_exc()
        failed_file_json_object = json.dumps(str(file))
        error_filepath = LOCAL_VIDEO_DIR + "_whisper_errors.jsonl"
        if not os.path.exists(error_filepath):
          pathlib.Path(error_filepath).touch()
        with jsonlines.open(error_filepath, mode='a') as writer:
          writer.write({"video_filepath": failed_file_json_object, "error": str(e)})

      # one file done
      print(
          f"⏰ Time to Whisper the file: {(time.monotonic() - start)/60:.2f} minutes\nVideo filesize: {os.path.getsize(file)/1e6:.2f} MB\n")


def write_error(file):
  failed_file_json_object = json.dumps(str(file))
  empty_filepath = LOCAL_VIDEO_DIR + "_whisper_empty.jsonl"
  if not os.path.exists(empty_filepath):
    pathlib.Path(empty_filepath).touch()
  with jsonlines.open(empty_filepath, mode='a') as writer:
    writer.write({"video_filepath": failed_file_json_object})


def find_files(directory: os.PathLike):
  filepaths = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      filepaths.append(os.path.join(root, file))
  return filepaths


@ray.remote(num_cpus=1)
def DEPRICATED_add_to_dataset(whisper_one_video_results):
  print("Entering add_to_dataset")
  ds = dl.load(WHISPER_RESULTS_DATASET_PATH)
  # try catch at level of a whole video...
  try:
    with ds:
      for segment in whisper_one_video_results:
        metadata = {
            "start": str(segment["start"]),
            "end": str(segment["end"]),
            "segment_word_list": segment["segment_word_list"],
            "segment_index": str(segment["segment_index"]),
            "total_segments": str(segment["total_segments"])
        }
        ds.caption.append(segment['caption'])
        ds.video_filename.append(segment["video_filename_name"])
        ds.video_filepath.append(segment["video_filepath"])
        ds.segment_metadata.append(dict(metadata))
  except Exception as e:
    print("-------------------------------START OF ERROR-------------------------------")
    pprint.pprint(whisper_one_video_results)
    print("^^^ FULL WHISPER RESULTS ^^^")
    print(f"Error in add_to_dataset, with file {segment['video_filepath']}. Error: {e}")
    print(f"Data being added during error:")
    pprint.pprint(segment)
    print(traceback.print_exc())
  finally:
    print(ds.summary())
    ds.flush()


def main():
  """ MAIN """
  # ray.shutdown()
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard=False, ignore_reinit_error=True)
  print_cluster_stats()
  start = time.monotonic()
  files = find_files(INPUT_VIDEOS_PATH)
  print(f"⏰ Time to collect input video files: {(time.monotonic() - start):.2f} seconds")

  # filter out bad files (.vtt and .wav, and .json) Anything other than webm and mp4?
  files = [str(file) for file in files if not str(file).endswith(('.txt', '.vtt', 'json'))]
  print("After filtering -- Number of files:", len(files))
  if os.path.exists(WHISPER_RESULTS_DATASET_PATH):
    # Filter files that we need to process
    print(f'Number of files before filtering completed files {len(files)}')
    ds_completed = dl.load(WHISPER_RESULTS_DATASET_PATH)
    ds_completed.summary()
    try:
      completed_videos = set()
      for index, data_row in enumerate(ds_completed):
        # try:
        completed_videos.add(data_row.video_filepath.data()['value'])
        # except Exception as e:
        #   print("Datalake unable to load index", index, "error is", e)
      print("⭐️😁 num videos already processed:", len(list(completed_videos)))
      files = set(files) - completed_videos
      files = list(files)
      print(f'Number of files after filtering completed files {len(files)}')
    except Exception as e:
      print("Error", e)
      print("There is an empty database already created")
  else:
    # Create completed files database
    ds = dl.empty(WHISPER_RESULTS_DATASET_PATH, overwrite=True)
    # todo: change to chunk_compression -- NOOO Chunk has BUGS as confirmed by devs. Use common compression types only.
    # don't use ANY compression on json fields. Always buggy.
    with ds:
      ds.create_tensor('caption', htype='text', dtype=str, sample_compression=None)
      ds.create_tensor('segment_metadata', htype='text', dtype=str, sample_compression=None)
      ds.create_tensor('video_filename', htype='text', dtype=str, sample_compression=None)
      ds.create_tensor('video_filepath', htype='text', dtype=str, sample_compression=None)

  # split files into batches
  if NUM_PARALLEL_INSTANCES == 1:
    batches = [files]
  else:
    batches = list(more_itertools.divide(NUM_PARALLEL_INSTANCES, files))
  # print batch stats
  print("Num batches: ", len(batches))
  assert len(batches) == (NUM_PARALLEL_INSTANCES), "there is supposed to be one Ray thread per batch"

  if not os.path.isdir(LOCAL_VIDEO_DIR + "_wav"):
    os.mkdir(LOCAL_VIDEO_DIR + "_wav")

  print("Starting parallel batches")
  parallel_whisper = ParallelWhisper.remote()

  # slow start so GPUs can spin up & dynamically assign processes to most free GPUs.
  futures = []
  for batch in batches:
    time.sleep(5)
    futures.append(parallel_whisper.parallel_caption_extraction.remote(batch))
  all_done = ray.get(futures)
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
