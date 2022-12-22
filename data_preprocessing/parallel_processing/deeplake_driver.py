import time
import pprint
import traceback
from collections import deque

# fancy
import deeplake as dl
import ray
from ray.util.queue import Queue
import asyncio


@ray.remote(concurrency_groups={"upload_driver": 1, "single_thread_io": 1, "parallel_ingest": 4})
class DeeplakeManager():
  def __init__(self, database_path=None, upload_queue=None):
    ray.init('auto', ignore_reinit_error=True) # todo: connect to existing ray cluster...
    
    # open and persist DB connection
    self.ds = dl.load(database_path)
    print(self.ds.summary())
    
    self.upload_queue = upload_queue
    
    # runs forever in background
    ray.get(self._start_whisper_upload_driver())
  
  @ray.method(concurrency_group="parallel_ingest") 
  def whisper_results_to_deeplake(self, whisper_one_video_results):
    '''
    Users call this function to upload data to Deeplake database. Upload is done in background.
    param: whisper_one_video_results: list of dicts, each dict is a segment
    '''
    print("👉 adding to upload queue...")
    # self.whisper_results_upload_queue.put(whisper_one_video_results)
    pass
  
  ############################
  ##### INTERNAL METHODS #####
  ############################
  @ray.method(concurrency_group="upload_driver") 
  def _start_whisper_upload_driver(self):
    print("Started the forever upload driver...")
    while True: # continuously check upload queue
      # self._whisper_results_to_deeplake()  # todo: <-- this is good for Whisper
      # todo: figure out how to call the right function per pre-processor
      self._text_encode_results_to_deeplake()
      time.sleep(3) # wait before checking again
  
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
      print(f"Error in _text_encode_results_to_deeplake(). Error: {e}")
      print(f"Data being added during error:")
      pprint.pprint(caption_embed_dict)
      print(traceback.print_exc())
  
  @ray.method(concurrency_group="single_thread_io")
  def _whisper_results_to_deeplake(self):
    print("👀 checking for things to upload...")
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
      print(f"Error in add_to_dataset, with file {segment['video_filepath']}. Error: {e}")
      print(f"Data being added during error:")
      pprint.pprint(segment)
      print(traceback.print_exc())
  
  def get_already_processed_files(self):
    # return list of files that have already been processed
    pass
  
  
if __name__ == "__main__":
  pass