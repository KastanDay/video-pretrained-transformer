import concurrent.futures
import inspect
import os
import pathlib
import time

import cv2
import lovely_tensors as lt
import numpy as np
import torch
from decord import (  # best random-access video reader all the lands.
    VideoReader, cpu)
from PIL import Image
from termcolor import colored
from transformers import CLIPProcessor, CLIPVisionModel, logging

lt.monkey_patch()

# import skvideo.io
# import imageio.v3 as iio
# import av
# import clip

# pyright: reportPrivateImportUsage=false
# pyright: reportOptionalMemberAccess=false
# ^^ due to not understanding ray

# suppress: Some weights of the model checkpoint at google/flan-t5-large were not used when initializing model.
# This is expected because we're initializing the encoder-only. So the decoder weights are not used.
logging.set_verbosity_error()

### GLOBALS SET ME 😁 ###
MODEL_SIZE = 'openai/clip-vit-large-patch14-336'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.
FRAME_SIZE_DIMENSION = 336
NUM_FRAMES_TO_SAVE_PER_SEGMENT = 1


class ClipEncoder:

  def __init__(self, debug=False, num_frames_per_segment=1):
    self.debug = debug
    self.num_frames_per_segment = num_frames_per_segment

    # Load the model
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {self.device}...")

    self.clip = CLIPVisionModel.from_pretrained(MODEL_SIZE).to(self.device)
    self.clip_preprocess = CLIPProcessor.from_pretrained(MODEL_SIZE)

    # self.clip, self.clip_preprocess = clip.load(MODEL_SIZE, self.device)
    if '336' in MODEL_SIZE:
      assert FRAME_SIZE_DIMENSION >= 336, "Frame size must be at least 336px (by 336) for ViT-L/14@336px"

    if self.debug:
      print(f"Done setting up CLIP...")

  def run_clip_one_batch(self, batch_of_100_samples):
    '''
    batch_of_100_samples format. It's always 100 segments!
    batch_of_100_samples = {
    '<full_video_filepath>':  [ {'timestamp': midpoint, 'db_index': idx} ],   # list of [times & midpoints] (1 per frame)
    '<full_video_filepath_2>':[ {'timestamp': midpoint, 'db_index': idx} ],  
    }
    '''
    all_frames = []
    all_timestamps = []
    all_db_indexes = []
    for video_filepath, time_and_db_index_list in batch_of_100_samples.items():
      # can return None if any frame failes to extract from video...
      curr_timestamps = [segment_dict['timestamp'] for segment_dict in time_and_db_index_list]
      local_frames = extract_frames_from_video(video_filepath, curr_timestamps, use_multithreading=True)
      if not (None in local_frames):
        all_timestamps.extend([segment_dict['timestamp'] for segment_dict in time_and_db_index_list])
        all_db_indexes.extend([segment_dict['db_index'] for segment_dict in time_and_db_index_list])
        all_frames.extend(local_frames)
      else:
        print(f"🚨🚨 Warning (can happen occasionally): failed to extract frames for video {video_filepath}")
        print(f"len(local_frames) is {len(local_frames)}")

    # RUN CLIP
    all_pooled_clip_embeds, last_hidden_states = self.run_clip(all_frames)

    results_dict = {
        'frames': all_frames,
        'last_hidden_states': last_hidden_states,
        'pooled_clip_embeds': all_pooled_clip_embeds,
        'timestamps': all_timestamps,
        'db_indexes': all_db_indexes,
    }
    return results_dict

  def run_clip(self, all_frames):
    '''
    :param frames: list of np.ndarrays
    :returns: np.ndarrays
    '''
    start_time = time.monotonic()
    # optional improvement: send in a list of images instead. Just worried about convert_RGB in that case...
    image_inputs = self.clip_preprocess(images=all_frames, return_tensors="pt").to(self.device)

    if self.debug:
      print(f"⏰  Runtime of preprocessing: {(time.monotonic() - start_time):.2f} seconds")
      print("RIGHT before running clip 📸")
    start_time = time.monotonic()
    with torch.inference_mode():  # even faster than no_grad()
      outputs = self.clip(**image_inputs, output_hidden_states=True, return_dict=True)
      all_pooled_clip_embeds = outputs['pooler_output'].cpu().numpy()  # (batch_size, hidden_size). FloatTensor
      last_hidden_states = outputs['last_hidden_state'].cpu().numpy(
      )  # (batch_size, sequence_length, hidden_size). FloatTensor
    if self.debug:
      print(
          f"⏰ CLIP Runtime on {len(all_frames)*self.num_frames_per_segment} images: {(time.monotonic() - start_time):.2f} seconds"
      )
      print("Clip all_pooled_clip_embeds.shape:")
      print(all_pooled_clip_embeds.shape)
      print("Clip last_hidden_states.shape:")
      print(last_hidden_states.shape)

    return all_pooled_clip_embeds, last_hidden_states


'''
VIDEO PROCESSING ADAPTED FROM MERLOT RESERVE
https://github.com/rowanz/merlot_reserve/blob/main/mreserve/preprocess.py
'''


def extract_frames_from_video(video_file, times, use_multithreading=False, blackbar_threshold=32, max_perc_to_trim=.20):
  """
  Extracts multiple things from the video and even handles black bars
  :param video_file: what we are loading
  :param times: timestamps to use
  :param use_multithreading: Whether to use multithreading
  :param use_rgb whether to use RGB (default) or BGR
  :param blackbar_threshold: Pixels must be this intense for us to not trim
  :param max_perc_to_trim: Will trim 20% by default of the image at most in each dimension
  :return: Frames that are trimmed to not have any black bars
  """
  # I had terrible "DecodeError" problems with num_threads > 1.. no idea why, no progress on GH issues: https://github.com/dmlc/decord/issues/124
  vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
  fps = vr.get_avg_fps()
  # timestamp in seconds -> frame_index
  frame_indexes = []
  for t in times:
    frame_indexes.append(int(t * fps))
  frames = vr.get_batch(frame_indexes).asnumpy()
  # print(f"⏰ Time to get {len(frame_indexes)} frames: {(time.monotonic() - start_time):.2f} seconds (time/frame = {((time.monotonic() - start_time)/len(frame_indexes)):.2f} sec)")
  return frames
  y1, y2, x1, x2 = _detect_black_bars_from_video(frames,
                                                 blackbar_threshold=blackbar_threshold,
                                                 max_perc_to_trim=max_perc_to_trim)
  return frames[:, y1:y2, x1:x2]

  # Original method. Over-complicated ANDDD slower. Fuck ffmpeg.

  def _extract(i):
    #   return i, extract_single_frame_from_video(video_file, times[i]) # todo: pass in video framerate.. for use in imageio.v3
    return i, kas_extract_single_frame_from_video(
        video_file, times[i], video_framerate)  # todo: pass in video framerate.. for use in imageio.v3

  if not use_multithreading:
    frames = [_extract(i)[1] for i in range(len(times))]
  else:
    frames = [None for t in times]
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
      submitted_threads = (executor.submit(_extract, i) for i in range(len(times)))
      for future in concurrent.futures.as_completed(submitted_threads):
        try:
          i, img = future.result()
          frames[i] = img
        except Exception as exc:
          print("❌❌❌❌ Oh no. Failed to extract frame using multithreading {}".format(str(exc)), flush=True)

  # If any frames fail, fail the WHOLE video... lame?
  if any([x is None for x in frames]):
    print(f"❌❌❌❌ Fail on {video_file}", flush=True)
    print(colored(f"❌ Fail extracing frames on {video_file}, in function {inspect.currentframe().f_code.co_name}",
                  "red",
                  attrs=["reverse", "bold"]),
          flush=True)
    return None

  frames = np.stack(frames)
  y1, y2, x1, x2 = _detect_black_bars_from_video(frames,
                                                 blackbar_threshold=blackbar_threshold,
                                                 max_perc_to_trim=max_perc_to_trim)

  return frames
  print("Right before returning frames")
  return frames[:, y1:y2, x1:x2]


def _detect_black_bars_from_video(frames, blackbar_threshold=16, max_perc_to_trim=.2):
  """
    :param frames: [num_frames, height, width, 3]
    :param blackbar_threshold: Pixels must be this intense for us to not trim
    :param max_perc_to_prim: Will trim 20% by default of the image at most in each dimension
    :return:
    """
  # Detect black bars
  has_content = frames.max(axis=(0, -1)) >= blackbar_threshold
  h, w = has_content.shape

  y_frames = np.where(has_content.any(1))[0]
  if y_frames.size == 0:
    print("Oh no, there are no valid yframes")
    y_frames = [h // 2]

  y1 = min(y_frames[0], int(h * max_perc_to_trim))
  y2 = max(y_frames[-1] + 1, int(h * (1 - max_perc_to_trim)))

  x_frames = np.where(has_content.any(0))[0]
  if x_frames.size == 0:
    print("Oh no, there are no valid xframes")
    x_frames = [w // 2]
  x1 = min(x_frames[0], int(w * max_perc_to_trim))
  x2 = max(x_frames[-1] + 1, int(w * (1 - max_perc_to_trim)))
  return y1, y2, x1, x2


def DEPRICATED_kas_extract_single_frame_from_video(video_file, timestamp_sec, video_fps):
  start_time = time.monotonic()
  frame_number = int(timestamp_sec * video_fps)
  frame = iio.imread(
      f"{video_file}",
      index=frame_number,
      plugin="pyav",
  )
  print(f"⏰ Time to extract 1 frame: {(time.monotonic() - start_time):.2f} seconds")
  return frame


def DEPRICATED_extract_single_frame_from_video(video_file, t):
  ## DEPRICATED -- prefer kas_extract_single_frame_from_video
  ## My version might be faster than  ffmpeg via skvideo interface
  # IDK, I'm just sick of ffmpeg trouble. This av seems more reliable for standard-complexity jobs.
  # Reserve FFMPEG usage for really difficult/niche problems.
  """
  Reads the video, seeks to the given second option
  :param video_file: input video file
  :param t: where 2 seek to
  :return: the frame at that timestep.
  """
  print("In extract SINGLE FRAME frame from video")
  start_time = time.monotonic()

  timecode = '{:.3f}'.format(t)
  print("timecode", timecode)
  # https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax. Time in seconds (decimal-notation) by default.
  input_dict = {'-ss': timecode, '-threads': '1'}

  print("Before skvideo reader")
  reader = skvideo.io.FFmpegReader(
      video_file,
      inputdict=input_dict,
      # outputdict={'-r': '1', '-q:v': '2', '-pix_fmt': 'rgb24', '-frames:v': '1'},
      outputdict={
          '-q:v': '2',
          '-pix_fmt': 'rgb24',
          '-frames:v': '1'
      },
      verbosity=0,
  )
  print("After skvideo reader")
  try:
    frame = next(iter(reader.nextFrame()))
  except StopIteration:
    frame = None
  print(f"⏰ Runtime: {(time.monotonic() - start_time):.3f} seconds")
  print("right before return in SINGLE frame")
  return frame