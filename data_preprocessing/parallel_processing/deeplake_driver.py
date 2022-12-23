import time
import pprint
import traceback
from collections import deque

# fancy
import deeplake as dl
import ray
from ray.util.queue import Queue
import asyncio
import inspect

@ray.remote(concurrency_groups={"upload_driver": 1, "single_thread_io": 1, "parallel_ingest": 4})
class DeeplakeManager():
  def __init__(self, preprocessor_type=None, database_path=None, upload_queue=None):
    assert preprocessor_type in ['whisper', 'clip', 'text-encode'], "only these modes are supported. Due to custom upload function for each."
    
    ray.init('auto', ignore_reinit_error=True) # todo: connect to existing ray cluster...
    
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
    assert preprocessor_type in ['whisper', 'clip', 'text-encode'], "only these modes are supported. Due to custom upload function for each."
    if preprocessor_type == 'whisper':
      while True: # continuously check upload queue
        self._whisper_results_to_deeplake()
        time.sleep(3) # wait before checking again
    elif preprocessor_type == 'clip':
      while True: 
        self._clip_encode_results_to_deeplake()
        time.sleep(3)
    elif preprocessor_type == 'text-encode':
      while True: # continuously check upload queue
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
          print("Queue size:", self.upload_queue.qsize())
          results = self.upload_queue.get(block=True)
          # loop over one segment at a time
          for all_frames, last_hidden_states, all_pooled_clip_embeds, timestamp, db_index in zip(results['frames'], results['last_hidden_states'], results['pooled_clip_embeds'], results['timestamps'], results['db_indexes']):
            self.ds.clip_pooled_embedding[db_index] = all_pooled_clip_embeds
            self.ds.clip_last_hidden_states[db_index] = last_hidden_states
            self.ds.frames[db_index] = all_frames
            # update metadata
            metadata = self.ds.segment_metadata[db_index].data()['value']
            metadata['clip_embedding'] = True
            metadata['frame_timestamp_sec'] = timestamp
            self.ds.segment_metadata[db_index] = metadata
        print("✅ SUCCESSFULLY uploaded a batch of segments to Deeplake! ✅")
        print(self.ds.summary())
        print("Queue size (should be zero):", self.upload_queue.qsize())
        self.ds.flush()
    except Exception as e:
      print("-----------❌❌❌❌------------START OF ERROR-----------❌❌❌❌------------")
      pprint.pprint(results)
      print("^^^ FULL clip-embed RESULTS ^^^")
      print(f"Error occurred at index: 👉 {db_index} 👈")
      print(f"Error in {inspect.currentframe().f_code.co_name}: {e}")
      print(traceback.print_exc())
      print("Testing getting the name of the curr function: ", print(inspect.currentframe().f_code.co_name))
  
  @ray.method(concurrency_group="single_thread_io")
  def _text_encode_results_to_deeplake(self):
    # print("👀 checking for things to upload...")
    try:
      with self.ds:
        while self.upload_queue.qsize() > 0:
          print("Queue size:", self.upload_queue.qsize())
          print("👉⬆️ STARTING AN ACTUAL UPLOAD... ⬆️👈")
          caption_embed_dict = self.upload_queue.get(block=True)
          self.ds.caption_embedding[caption_embed_dict['db_index']] = caption_embed_dict['last_hidden_states']
          print("✅ SUCCESSFULLY finished uploading to Deeplake! ✅")
          print(self.ds.summary())
    except Exception as e:
      print("-----------❌❌❌❌------------START OF ERROR-----------❌❌❌❌------------")
      pprint.pprint(caption_embed_dict)
      print("^^^ FULL text-embed RESULTS ^^^")
      print(f"Error in {inspect.currentframe().f_code.co_name}: {e}")
      print(f"Data being added during error:")
      pprint.pprint(caption_embed_dict)
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
              self.ds.caption.append(segment['caption'])
              self.ds.video_filename.append(segment["video_filename_name"])
              self.ds.video_filepath.append(segment["video_filepath"])
              self.ds.segment_metadata.append(dict(metadata))
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
  
  def get_already_processed_files(self):
    # return list of files that have already been processed
    pass
  
  
if __name__ == "__main__":
  pass