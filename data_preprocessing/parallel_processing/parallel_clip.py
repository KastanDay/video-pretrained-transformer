import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first

import sys
# sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing") # DELTA
# sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing") # HAL
# sys.path.append("/home/kastan/thesis/video-pretrained-transformer/data_preprocessing/parallel_processing/")  # Kastan server
# from data_preprocessing import ClipPreprocessor

import json
import numpy as np
from PIL import Image
import pathlib
from pathlib import Path
import glob
import argparse
import jsonlines
import json
import time
import deeplake as dl
import tqdm
import sys
import ray
import more_itertools
import traceback
import pprint

global NUM_THREADS
global NUM_CPU_CORES
global NUM_GPUS
global GPU_PER_PROCESS

# datasets
BATCH_NAME                   = 'parallel_15'
# WHISPER_RESULTS_DATASET_PATH = f'/mnt/storage_ssd/whisper_results_{BATCH_NAME}'
WHISPER_RESULTS_DATASET_PATH = f'/mnt/storage_ssd/dummy_clip_results_parallel_15'
# TEST_CLIP_RESULTS_DATASET    = f'/mnt/storage_ssd/clip_results_{BATCH_NAME}'


# THIS is GREAT balance on delta GPU, 4X GPU with clip running
# NUM_THREADS = 20      # Number of parallel processes (limited by DRAM and SRAM)
# NUM_CPU_CORES = 64    # Numer of available physical cores to use (use max!)
# NUM_GPUS = 4          # Number of physical GPUs to use (use max)
# GPU_PER_PROCESS = 1/5 # threads per GPU, limited by OOM errors while also maximizing spread. 

NUM_THREADS = 2      # Number of parallel processes (limited by DRAM and SRAM)
NUM_CPU_CORES = 8    # Numer of available physical cores to use (use max!)
NUM_GPUS = 1          # Number of physical GPUs to use (use max)
GPU_PER_PROCESS = 1 # threads per GPU, limited by OOM errors while also maximizing spread. 

@ray.remote(num_cpus=8, num_gpus=1) # 2 and 1/4
def parallel_clip(batch_of_100_samples):
    start = time.monotonic()
    # sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing/parallel_processing")  # Kastan server
    sys.path.append("/home/kastan/thesis/video-pretrained-transformer/data_preprocessing/parallel_processing/")  # Kastan server
    # sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing") # Delta
    
    from clip_preprocessing import ClipPreprocessor
    my_clip_preprocesser = ClipPreprocessor(debug=True)
    print('in parallel clip')
    print(batch_of_100_samples)
    
    for sample_dict in batch_of_100_samples:
        try:
            print(f"Starting file {sample_dict}")
            
            # TODO: WAIT WHY 202???? 
            # shape = 100 <i.e. batch_size>, 360, 202, 3
            # TODO: add FRAMES to our deeplake dataset, too. Need those frames.
            all_frames, all_pooled_clip_embeds, all_timestamps, all_db_indexes = my_clip_preprocesser.run_clip_one_video(sample_dict)
            hidden_states = None
            dataset_future = add_to_dataset.remote(all_pooled_clip_embeds, all_timestamps, all_db_indexes)
            ray.get(dataset_future)
            
            print(f"✅ Time to CLIP the video: ⏰ {(time.monotonic() - start)/60:.2f} minutes")
        except KeyError as ke:
            pass # ignore for testing
            # print("Missing video file: ", file_stem)
            # todo: write to file just like Daniel.
        except Exception as e:
            # write failed files to jsonlines
            print(f"Error during CLIP: {e}\n \___Traceback exception:")
            traceback.print_exc()
            
            failed_file_json_object = json.dumps(str(sample_dict))
            error_filepath = WHISPER_RESULTS_DATASET_PATH + "_clip_errors.jsonl"
            if not os.path.exists(error_filepath):
                pathlib.Path(error_filepath).touch()
            with jsonlines.open(error_filepath, mode='a') as writer:
                writer.write({"video_filepath": failed_file_json_object, "error": str(e)}) 
        start = time.monotonic()

    # one file done        
    # print(f"⏰ Time to Whisper the file: {(time.monotonic() - start)/60:.2f} minutes\nVideo filesize: {os.path.getsize(file)/1e6:.2f} MB\n")
    start = time.monotonic()

import traceback
@ray.remote(num_cpus=1)
def add_to_dataset(all_frames, all_pooled_clip_embeds, all_timestamps, all_db_indexes):
  # todo: more to be done...
  ds = dl.load(WHISPER_RESULTS_DATASET_PATH)
  print("IN THE ADD_TO_DATASET")
  try:
    with ds:
      for pooled_clip_embedding, timestamp, db_index in zip(all_pooled_clip_embeds, all_timestamps, all_db_indexes):
        print("About to ADD TO DATASET BY INDEX >>>> 👇👇")
        ds.pooled_clip_embedding[db_index] = pooled_clip_embedding.reshape((768))
        ds.clip_hidden_states[db_index] = pooled_clip_embedding.reshape((768)) # todo: update
        ds.all_frames[db_index] = all_frames
        
        metadata = ds.segment_metadata[db_index].data()['value']
        metadata['clip_embedding'] = True
        metadata['frame_timestamp_sec'] = timestamp
        ds.segment_metadata[db_index] = metadata
  except Exception as e:
    print(f"Error {e}")
    print(traceback.print_exc())
  finally:
    print(ds.summary(), flush=True)
    ds.flush()
    
      
def main():
  """ MAIN """
  # ray.shutdown()
  ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
  print_cluster_stats()
  
  # load dataset  
  clip_outputs_ds = dl.load(WHISPER_RESULTS_DATASET_PATH) 
  # create tensor if doesn't exist yet
  if 'pooled_clip_embedding' not in clip_outputs_ds.tensors.keys():
    # CLIP produces FP32 embeddings.
    clip_outputs_ds.create_tensor('pooled_clip_embedding', exist_ok=True, htype='image', dtype=np.float32, sample_compression='lz4')
  if 'clip_hidden_states' not in clip_outputs_ds.tensors.keys():
    # CLIP produces FP32 embeddings.
    clip_outputs_ds.create_tensor('clip_hidden_states',    exist_ok=True, htype='image', dtype=np.float32, sample_compression='lz4')
  if 'frames' not in clip_outputs_ds.tensors.keys():
    # Not sure of this dtype?? for PIL Image().fromarray()
    clip_outputs_ds.create_tensor('frames',            exist_ok=True, htype='image', sample_compression='jpeg') # not sure of datatype
  print(clip_outputs_ds.summary(), flush=True)
  
  # fill with zeros, so we can index and populate it.
  with clip_outputs_ds:
    for _ in range(clip_outputs_ds.max_len):
      clip_outputs_ds.pooled_clip_embedding.append(None) # np.float16(-1))
      clip_outputs_ds.clip_hidden_states.append(None)  # np.float16(-1))
      clip_outputs_ds.clip_hidden_states.append(None)
  clip_outputs_ds.flush()
  print(clip_outputs_ds.summary(), flush=True)
  
  # make groups of 100 samples for Clip
  # struct format:
  # hundred_samples = {
  #   '<full_video_filepath>': [], # list of timestamp midpoints (1 per frame)
  #   '<full_video_filepath_2>': [], # list of timestamp midpoints (1 per frame)
  # }
  hundred_samples = {}
  batches_of_100_samples = []
  ds = dl.load(WHISPER_RESULTS_DATASET_PATH)
  
  # TODO: Re-write the batching to just make one huge list, then use itertools to batch.
  
  print("⚠️ Only using 100 samples for testing")
  ds = ds[:5]
  for idx in range(ds.max_len):
    # todo: see if clip exists *for EACH segment*
    # todo: if sample.pooled_clip_embedding.data()['value'] is not None:......
    
    sample = ds[idx]
    seg_start_time = sample['segment_metadata'].data(fetch_chunks=True)['value']['start']
    seg_end_time   = sample['segment_metadata'].data()['value']['end']
    midpoint = (seg_end_time + seg_start_time) / 2
    
    # add to dict {'<filepath>': [<frame_timestamp>, ...]}
    video_filepath = sample['video_filepath'].data(fetch_chunks=True)['value']
    if video_filepath not in hundred_samples.keys():
      hundred_samples[video_filepath] = []
    hundred_samples[video_filepath].append( {'timestamp': midpoint, 'db_index': idx} )
    
    # make batches of 100 samples
    if len(hundred_samples[video_filepath]) == 100:
      batches_of_100_samples.append(hundred_samples)
      hundred_samples = {}
  if len(batches_of_100_samples) == 0 and len(hundred_samples) > 0:
    batches_of_100_samples.append(hundred_samples)
  else:
    assert False, print("Error: batches_of_100_samples is empty. nothing to process...")
    
  
  # for i, sample in enumerate(ds):
  #   print(sample['caption'].data())
  #   print(sample['video_filepath'].data())
  #   pprint.pprint(sample['segment_metadata'].data()['value']['start'])
  #   pprint.pprint(sample['segment_metadata'].data()['value']['end'])
  #   pprint.pprint(sample['segment_metadata'].data()['value'])
  
  ''' Idea for how to filter. But porbably better if I check for existance.'''
  # files_to_process = set()
  # for i, sample in enumerate(video_filepaths):
  #   try: 
  #     pooled_clip_embedding = sample.pooled_clip_embedding.data()['value']
  #   except Exception as e:
  #     print(e)
  #     # no clip, add to list of files to process
  #     files_to_process.add(sample.video_filepath.data()['value'])
  # print(f"Seconds to iterate over dataset and check for existing CLIP results: {time.time() - start:.2f}")
  # print("Number of files:", len(files_to_process))
  
  
  # DO BATCHING
  if NUM_THREADS == 1:
    batches = [batches_of_100_samples]
  else:
    batches = list(more_itertools.divide(NUM_THREADS, batches_of_100_samples))
  print("Num batches: ", len(batches))
  print(len(batches), " should equal num threads: ", NUM_THREADS)
  assert len(batches) == (NUM_THREADS)

  print("Starting parallel batches")
  all_result_futures = [parallel_clip.remote(batch) for itr, batch in enumerate(batches)]
  
  all_done = ray.get(all_result_futures)

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

if __name__ == '__main__':
    main()