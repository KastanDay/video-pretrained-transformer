# from collections import deque
# import asyncio
import inspect
import json
import pathlib
import pprint
import time
import traceback

import deeplake as dl
import ray
import tqdm
from ray.util.queue import Queue
# fancy
from termcolor import colored

# pyright: reportGeneralTypeIssues=false
# ^^ due to not understanding deeplake
# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray


@ray.remote(concurrency_groups={"upload_driver": 1, "single_thread_io": 1, "parallel_ingest": 4})
class DeeplakeManager():

  def __init__(self, preprocessor_type=None, database_path=None, upload_queue=None):
    assert preprocessor_type in ['whisper', 'clip', 'text-encode'
                                ], "only these modes are supported. Due to custom upload function for each."

    ray.init('auto', ignore_reinit_error=True)  # todo: connect to existing ray cluster...

    # open and persist DB connection
    self.ds = dl.load(database_path)
    print(self.ds.summary())

    self.upload_queue = upload_queue

    # runs forever in background
    ray.get(self._start_upload_driver(preprocessor_type))

  ############################
  ##### INTERNAL METHODS #####
  ############################
  @ray.method(concurrency_group="upload_driver")
  def _start_upload_driver(self, preprocessor_type):
    '''
    The __init__() calls this function to upload data to Deeplake database. Upload is done in background.
    '''
    print("Started the forever upload driver...")
    assert preprocessor_type in ['whisper', 'clip', 'text-encode'
                                ], "only these modes are supported. Due to custom upload function for each."
    if preprocessor_type == 'whisper':
      while True:  # continuously check upload queue
        self._whisper_results_to_deeplake()
        time.sleep(3)  # wait before checking again
    elif preprocessor_type == 'clip':
      while True:
        self._clip_encode_results_to_deeplake()
        time.sleep(3)
    elif preprocessor_type == 'text-encode':
      while True:  # continuously check upload queue
        self._text_encode_results_to_deeplake()
        time.sleep(3)

  @ray.method(concurrency_group="single_thread_io")
  def _clip_encode_results_to_deeplake(self):
    '''
    shape of results dict:
    Each value is a list of equal length, everything is length 100 for GPU-memory reasons.
    results = {
        'frames': all_frames,
        'last_hidden_states': last_hidden_states,
        'pooled_clip_embeds': all_pooled_clip_embeds,
        'timestamps': all_timestamps,
        'db_indexes': all_db_indexes,
    }
    '''
    try:
      with self.ds:
        while self.upload_queue.qsize() > 0:
          print("🔁 Queue size:", self.upload_queue.qsize())
          start_time = time.monotonic()
          results = self.upload_queue.get(block=True)
          print(f"🔁⏰ time to pull one result from queue: {(time.monotonic() - start_time):.2f} seconds")
          # todo: update via slice instead of iterating. Probably faster.
          # https://docs.deeplake.ai/en/latest/deeplake.core.tensor.html#deeplake.core.tensor.Tensor.__setitem__
          # Maybe occasionally rechunk: ds.rechunk(num_workers , scheduler)
          # Todo: WATCH OUT this is some really experimental shit right here.
          # todo: test SLICING the db inserts.
          start_time = time.monotonic()

          # print("Inner frames shape", (results['frames'][0].shape))
          for frame in results['frames']:
            if frame.shape != (360, 640, 3):
              print(frame.shape)
          # print("Inner pooled shape", (results['pooled_clip_embeds'][0].shape))
          # print("Inner last_hidden shape", results['last_hidden_states'].shape)
          # print("Inner last_hidden shape", results['last_hidden_states'][0].shape)

          first_idx, last_idx = results['db_indexes'][0], results['db_indexes'][-1]
          last_idx = last_idx + 1  # Is this right?
          assert check_continuity(
              results['db_indexes']), print("db_inxexes must be continuous. This batch is not contiguous in Deeplake.")
          assert last_idx - first_idx == (len(results['frames'])), print(
              f"Length of frames {len(results['frames'])} must equal {last_idx}-{first_idx} = {last_idx - first_idx}")
          self.ds.clip_pooled_embedding[first_idx:last_idx] = results['pooled_clip_embeds']
          self.ds.clip_last_hidden_states[first_idx:last_idx] = results['last_hidden_states']
          self.ds.frames[first_idx:last_idx] = results['frames']
          self.ds.timestamp[first_idx:last_idx] = results['timestamps']
          print(f"⬆️⏰ Time to upload one batch: {(time.monotonic() - start_time):.2f} seconds. Time per frame = {((time.monotonic() - start_time)/len(results['frames'])):.2f}") # yapf: disable
          # for all_frames, last_hidden_states, all_pooled_clip_embeds, timestamp, db_index in zip(results['frames'], results['last_hidden_states'], results['pooled_clip_embeds'], results['timestamps'], results['db_indexes']):
          # update metadata
          # metadata = self.ds.segment_metadata[db_index].data()['value']
          # metadata['clip_embedding'] = True
          # metadata['frame_timestamp_sec'] = timestamp
          # self.ds.segment_metadata[db_index] = metadata
        print("✅ SUCCESSFULLY caught up to Queue size for Deeplake! ✅")
        print(self.ds.summary())
        print("Queue size (should be zero):", self.upload_queue.qsize())
        self.ds.flush()
    except Exception as e:
      print("-----------❌❌❌❌------------START OF ERROR-----------❌❌❌❌------------")
      # pprint.pprint(results)
      print("^^^ FULL clip-embed RESULTS ^^^")
      print(f"Error occurred at index: 👉 {results['db_indexes']} 👈")
      print(f"Error in {inspect.currentframe().f_code.co_name}: {e}")
      print(traceback.print_exc())
      print("Testing getting the name of the curr function: ", print(inspect.currentframe().f_code.co_name))

  @ray.method(concurrency_group="single_thread_io")
  def _text_encode_results_to_deeplake(self):
    try:
      with self.ds:
        while self.upload_queue.qsize() > 0:
          print("👉⬆️ Upload queue size:", self.upload_queue.qsize(), "⬆️👈")
          print(" STARTING AN ACTUAL UPLOAD... ")
          start = time.monotonic()
          caption_embed_dict_list = self.upload_queue.get(block=True)
          for input_dict in caption_embed_dict_list:

            # True and true :)
            # print("Should be same shape: ", self.ds.caption_embedding[input_dict['db_index']].numpy().shape, input_dict['last_hidden_states'].shape)
            # print("Is continuous?", check_continuity([d['db_index'] for d in caption_embed_dict_list]))

            # todo: it's already fast enough, but if I want to...
            # todo: ...I confirmed it's okay to make this a batch-update operation (like clip), if db_index is continuous.
            # todo: index_list = [d['db_index'] for d in caption_embed_dict_list]
            # assert check_continuity(input_dict['db_index']), print("db_index must be continuous. This batch is not contiguous in Deeplake.")
            # assert last_idx-first_idx == (len(input_dict['frames'])), print(f"Length of frames {len(input_dict['frames'])} must equal {last_idx}-{first_idx} = {last_idx - first_idx}")

            self.ds.caption_embedding[input_dict['db_index']] = input_dict['last_hidden_states']
          print("✅ SUCCESSFULLY finished uploading to Deeplake! ✅")
          print(
              f"⬆️⬆️ Time to upload text-batch: {(time.monotonic() - start)/60:.2f} minutes. (time/segment): {((time.monotonic() - start)/len(caption_embed_dict_list)):.2f} sec"
          )
          print(self.ds.summary())
          self.ds.flush()
    except Exception as e:
      print("-----------❌❌❌❌------------START OF ERROR-----------❌❌❌❌------------")
      pprint.pprint(input_dict)
      print("^^^ FULL text-embed INPUTS ^^^")
      print(f"Error in {inspect.currentframe().f_code.co_name}: {e}")
      print(f"Data being added during error:")
      print(traceback.print_exc())

  @ray.method(concurrency_group="single_thread_io")
  def _whisper_results_to_deeplake(self):
    '''
    param: whisper_one_video_results: list of dicts, each dict is a segment
    '''
    try:
      while self.upload_queue.qsize() > 0:
        whisper_one_video_results = self.upload_queue.get(block=True)
        with self.ds:
          # loop over upload queue (best done here to keep the 'with' context manager open)
          print("👉⬆️ STARTING AN ACTUAL UPLOAD... ⬆️👈")
          for segment in whisper_one_video_results:
            metadata = {
                "start": str(segment["start"]),
                "end": str(segment["end"]),
                "segment_word_list": segment["segment_word_list"],
                "segment_index": str(segment["segment_index"]),
                "total_segments": str(segment["total_segments"])
            }
            # atomic append. All work or none get added.
            self.ds.append({
                'caption': segment['caption'],
                'video_filename': segment["video_filename_name"],
                'video_filepath': segment["video_filepath"],
                'segment_metadata': dict(json.dumps(metadata)),
            })
          print("✅ SUCCESSFULLY finished uploading to Deeplake! ✅")
          print(self.ds.summary())
        self.ds.flush()
    except Exception as e:
      print("-----------❌❌❌❌------------START OF ERROR-----------❌❌❌❌------------")
      pprint.pprint(whisper_one_video_results)
      print("^^^ FULL WHISPER RESULTS ^^^")
      print(f"Error in {inspect.currentframe().f_code.co_name}:, with file {segment['video_filepath']}. Error: {e}")
      print(f"Data being added during error:")
      pprint.pprint(segment)
      print(traceback.print_exc())


import pickle


def compress_and_delete_dataset(dataset_path, destructive=False):
  ''' After we finish processing a dataset, we should compress it once and for all. '''
  # todo: implement destructive mode... Just delete original, and rename the new one.

  in_ds = dl.load(dataset_path)
  p = pathlib.Path(dataset_path)
  print(str(p.name))
  outpath = pathlib.Path(p.parent / (str(p.name) + '_compressed'))
  print(f"Creating dataset at path: {outpath}")
  out_ds = dl.empty(outpath, overwrite=True)
  # ds.rechunk(scheduler='ray', num_workers=10) # no need
  import numpy as np
  with out_ds:
    out_ds.create_tensor('caption', htype='text', dtype=str, sample_compression='lz4')
    out_ds.create_tensor('caption_embedding', htype='image', dtype=np.float16, sample_compression='lz4')
    out_ds.create_tensor('clip_last_hidden_states', htype='image', dtype=np.float32, sample_compression='lz4')
    out_ds.create_tensor('clip_pooled_embedding', htype='image', dtype=np.float32, sample_compression='lz4')
    out_ds.create_tensor('frames', htype='image', dtype=np.uint8, sample_compression='jpeg')
    out_ds.create_tensor('segment_metadata', htype='text', dtype=str, sample_compression='lz4')
    out_ds.create_tensor('timestamp', htype='generic', dtype=float, sample_compression='lz4')
    out_ds.create_tensor('video_filename', htype='text', dtype=str, sample_compression=None)
    out_ds.create_tensor('video_filepath', htype='text', dtype=str, sample_compression=None)
    print("Created new ds")
    print(out_ds.summary())

  print(colored(f"👉 Start creating the new, compressed, dataset", "cyan", attrs=["reverse", "bold"]))
  total_errors = 0
  with out_ds:
    for sample in tqdm.tqdm(in_ds):

      # full_sample = {
      #   'caption': sample.caption.data()['value'],
      #   'caption_embedding': sample.caption_embedding.data()['value'],
      #   'clip_last_hidden_states': sample.clip_last_hidden_states.data()['value'],
      #   'clip_pooled_embedding': sample.clip_pooled_embedding.data()['value'],
      #   'frames': sample.frames.data()['value'],
      #   'segment_metadata': sample.segment_metadata.data()['value'],
      #   'timestamp': sample.timestamp.data()['value'],
      #   'video_filename': sample.video_filename.data()['value'],
      #   'video_filepath': sample.video_filepath.data()['value'],
      # }
      # print(full_sample)
      # my_pickled_object = pickle.dumps(full_sample)
      # my_pickled_object.a_dict = None

      # file_to_deeplake().eval(my_pickled_object, out_ds, num_workers=2)# scheduler='ray', num_workers=11)
      try:
        out_ds.append({
            'caption': sample.caption.data()['value'],
            'caption_embedding': sample.caption_embedding.data()['value'],
            'clip_last_hidden_states': sample.clip_last_hidden_states.data()['value'],
            'clip_pooled_embedding': sample.clip_pooled_embedding.data()['value'],
            'frames': sample.frames.data()['value'],
            'segment_metadata': sample.segment_metadata.data()['value'],
            'timestamp': sample.timestamp.data()['value'],
            'video_filename': sample.video_filename.data()['value'],
            'video_filepath': sample.video_filepath.data()['value'],
        })
      except Exception as e:
        total_errors += 1
        print(f"❌❌ Error {inspect.currentframe().f_code.co_name}: {e}")
        print(traceback.print_exc())

    out_ds.flush()
    print(out_ds.summary())
    print(colored(f"✅ Successfully created compressed dataset at path: {outpath}", "green", attrs=["reverse", "bold"]))
    print(f"Total errors: {total_errors}")


# @dl.compute
# def file_to_deeplake(sample_in, sample_out):
#   sample_out.append(pickle.loads(sample_in))
#   return sample_out


def upload_dataset_to_hub(dataset_path: str, _optional_ds_name_on_hub: str = None):
  '''
  dataset_path: Filepath of local dataset. 
  _optional_ds_name_on_hub: Choose custom name for the dataset on the hub.
  '''
  if _optional_ds_name_on_hub:
    dl.deepcopy(dataset_path, f'hub://center-for-ai-innovation-ncsa/{_optional_ds_name_on_hub}')
  else:
    ds_name = pathlib.Path(dataset_path).name
    dl.deepcopy(dataset_path, f'hub://center-for-ai-innovation-ncsa/{ds_name}')


def upload_dataset_to_s3(dataset_path):
  raise NotImplementedError
  ds_name = pathlib.Path(dataset_path).name
  dl.deepcopy(dataset_path, f's3://handpicked_only/{ds_name}')


def check_continuity(my_list):
  '''
  https://stackoverflow.com/questions/48596542/how-to-check-all-the-integers-in-the-list-are-continuous
  '''
  return all(a + 1 == b for a, b in zip(my_list, my_list[1:]))


if __name__ == "__main__":
  pass