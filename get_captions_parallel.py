import json
from time import monotonic

# todo use ray for parallel downloading
import ray
import requests
import tensorflow as tf
from youtube_transcript_api import YouTubeTranscriptApi


def fake_to_real_ytid(ytid):
  assert len(ytid) == 4
  x = f"http://data.yt8m.org/2/j/i/{ytid[0:2]}/{ytid}.js"
  r = requests.get(x)
  if r.status_code != 200:
    raise Exception("Video was probably removed:", r.text)
    
  return r.text[10:21]

def get_captions(captions, start):
  end = start + 5
  relevant_captions = []
  for caption in captions:
    caption_start = caption['start']
    caption_end = caption_start + caption['duration']
    # Case 1: started before
    if (end > caption_start and end < caption_end) or (start > caption_start and start < caption_end):
      relevant_captions.append(caption['text'])
  return relevant_captions

def make_workflow_id(name: str) -> str:
  from datetime import datetime

  import pytz

  # Timezones: US/{Pacific, Mountain, Central, Eastern}
  # All timezones `pytz.all_timezones`. Always use caution with timezones.
  curr_time = datetime.now(pytz.timezone('US/Central'))
  return f"{name}-{str(curr_time.strftime('%h_%d,%Y@%H:%M'))}"

# save python dict to json file named 'captions.json'
def save_dict_to_json(captions, filename):
  with open(filename, 'w') as f:
    json.dump(captions, f)
  
@ray.remote(num_returns=1)
def get_captions_parallel(file):
  '''
  We expect about 10 objects per file.
  '''
  all_caption_dict = {}

  raw_dataset = tf.data.TFRecordDataset(file)
  # for raw_record in raw_dataset:
  for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    start_times = example.features.feature['segment_start_times'].int64_list.value
    fake_id = example.features.feature['id'].bytes_list.value[0].decode("utf-8")

    try:
      yt_id = fake_to_real_ytid(fake_id)
    except Exception as e:
      # probably 403 error, forbidden (removed video)
      continue

    try:
      url = YouTubeTranscriptApi.get_transcript(yt_id)
    except Exception as e:
      # probably no captions
      continue

    caption_list = []
    # daniel put your code here
    for index, start_time in enumerate(start_times):
      captions = get_captions(url, start_time)
      concat_captions = " ".join(captions)
      caption_list.append(concat_captions)
    
    # add current caption list to local dict, then pass that back to main thread.
    all_caption_dict.update( {yt_id: caption_list} )
    print('all_caption_dict', all_caption_dict)

  return all_caption_dict

if __name__ == "__main__":
  from time import monotonic
  ray.init()

  # TEST_FILE_LIMIT = 20

  # glob all .tfrecord files in ./YT_8M_data directory
  filenames = tf.io.gfile.glob('./YT_8M_data/*.tfrecord')
  # filenames = filenames[:TEST_FILE_LIMIT]

  start = monotonic()
  app_futures = []
  for filename in filenames:
    app_futures.append(get_captions_parallel.remote(filename))

  # DATA FORMAT: all_captions[video_id] = [caption0, caption1, ..., caption4]
  all_captions = {}
  num_no_caption = 0

  # wait for all tasks to finish
  for itr, future in enumerate(app_futures):
    captions_dict = ray.get(future)
    if captions_dict is None: 
      num_no_caption += 1
      continue 
    print("CAPTIONS HERE !!!\n\n", captions_dict)
    all_captions.update(captions_dict)
    # todo: save to json file every 200 * 10 videos
    if itr % 200 == 0:
      saved_outfilename = make_workflow_id(f"8M_all_captions_IN_PROG_{itr}__")
      save_dict_to_json(all_captions, saved_outfilename)
    
  saved_outfilename = make_workflow_id("8M_all_captions")
  save_dict_to_json(all_captions, saved_outfilename)

  print(all_captions)
  # print("👉 No caption ratio: ", num_no_caption / len(filenames))  
  print("👉 Total number of videos: ", len(all_captions))
  print(f"time taken: {(monotonic() - start):.2f}) seconds")
  
