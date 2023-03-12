import time

import deeplake as dl
import numpy as np
import psutil
import ray
from deeplake_driver import DeeplakeManager
from ray.util.queue import Queue

NUM_PARALLEL_PROCESSES = 2  # More than 2 and the uploader can't keep up.
NUM_CPU_CORES = psutil.cpu_count()  # Numer of available physical cores to use (use max!)
NUM_GPUS = 1  # Number of physical GPUs to use (use max)
GPU_PER_PROCESS = 1  # threads per GPU, limited by OOM errors while also maximizing spread.
BATCH_SIZE = 30  # 30 * 2 threads. good on 11GB


@ray.remote(concurrency_groups={"parallel_whisper_instances": NUM_PARALLEL_PROCESSES},
            num_cpus=0,
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

    self.work_queue = Queue()
    for batch in work_to_do_list:
      self.work_queue.put(batch)

  @ray.method(concurrency_group="parallel_whisper_instances")
  def parallel_clip_encode(self):
    '''
    Main function for parallel clip. 
    '''
    while self.work_queue.qsize() > 0:
      start = time.monotonic()
      batch = self.work_queue.get(block=True, timeout=10)
      time.sleep(5)
      print(f"⏰ One batch: clip-encoded {BATCH_SIZE} segments in {(time.monotonic() - start):.2f} sec.")

  def get_upload_queue_size(self):
    return self.upload_queue.qsize()

  def get_work_queue_size(self):
    return self.work_queue.qsize()


def main():

  parallel_encode = ParallelEncode.remote(work_to_do_list=[0] * 6)
  all_done = ray.get([parallel_encode.parallel_clip_encode.remote() for _ in range(NUM_PARALLEL_PROCESSES)])
  print("Len of all threads: ", len(all_done))
  print("👉 Completed compute.")

  # while ray.get(parallel_encode.get_upload_queue_size.remote()) > 0 or ray.get(parallel_encode.get_work_queue_size.remote()) > 0:
  #   print("Deeplake upload queue size", ray.get(parallel_encode.get_upload_queue_size.remote()))
  #   print("CLIP work queue size", ray.get(parallel_encode.get_work_queue_size.remote()))
  #   print("Still uploading files, sleeping 5 seconds..")
  #   time.sleep(5)
  # print("✅ All work and uploads should be done, exiting!")


if __name__ == "__main__":
  main()