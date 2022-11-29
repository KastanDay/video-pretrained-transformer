import os 
import json
import cv2
import numpy as np
import clip
import torch
from PIL import Image
import pathlib
from pathlib import Path
import glob
import argparse
import jsonlines
import json
import time
# import json_tricks # https://github.com/mverleg/pyjson_tricks
# import json_numpy

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # just for testing.
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3' # just for testing.


'''
⭐️ How to read the CLIP outputs ⭐️

itterate over arr_0 thru total_segments

path = '/scratch/bbki/kastanday/whisper/parallel_15_clip_output/LdMD528r6Xs_Jon\'s Daily Hustle_802_Lawn Care Equipment Setup Plans For 2021 - Upgrading Lawn Mowers.npz'
np_loaded = np.load(path, allow_pickle=True)
print(np_loaded)
np_loaded['arr_0'].item() # iterate here until `not .next()`

Docs: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
'''

'''
⭐️ How to read the CLIP outputs ⭐️

itterate over arr_0 thru total_segments

path = '/scratch/bbki/kastanday/whisper/parallel_15_clip_output/LdMD528r6Xs_Jon\'s Daily Hustle_802_Lawn Care Equipment Setup Plans For 2021 - Upgrading Lawn Mowers.npz'
np_loaded = np.load(path, allow_pickle=True)
print(np_loaded)
np_loaded['arr_0'].item() # iterate here until `not .next()`

Docs: https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed
'''


'''
INSTALL INSTRUCTIONS (STRICT dependencies, mostly due to Ray.):
conda create -n v3_clip_preprocessing_yt1b python=3.8.13 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install pandas "ray[default]==1.13.0" more_itertools jsonlines json_numpy pyarrow fastparquet pandas parquet ftfy regex tqdm git+https://github.com/openai/CLIP.git
conda install -c conda-forge -y git-lfs
cd into the git repo and run `git lfs install` and `git lfs pull`
(optional) pip install pretty_errors
'''

### GLOBALS SET ME 😁 ### 
MODEL_SIZE = 'ViT-L/14@336px'  # Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower.  
FRAME_SIZE_DIMENSION = 336
NUM_FRAMES_TO_SAVE_PER_SEGMENT = 1

def parse_cmd_line_args():
    """ Usage: 
    $ python data_preprocessing.py  --video_path /tmp/parallel_12
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help="path of video directory (not individual files). Ex: `--video_path /tmp/parallel_12`")
    
    # these are NOT NECESSARY, I construct it below. (using introduces error)
    # parser.add_argument('--audio_jsonl', type=str, help="path to json lines files containing audios")
    # parser.add_argument('--output_path', type=str, help="path of output directory") 
    args = parser.parse_args()
    
    # Convert: /tmp/parallel_10   into:   
    # output     = /tmp/parallel_10_clip_dataset.jsonl
    # whisper_in = /tmp/parallel_10_whisper_output.jsonl
    video_input_dir = pathlib.Path(args.video_path)
    args.output_path = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_clip_output.jsonl'))
    args.audio_jsonl = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_whisper_output.jsonl'))

    # validate input
    print("Using video path:", args.video_path)
    assert os.path.exists(args.video_path)
    assert os.path.exists(args.output_path)
    assert os.path.exists(args.audio_jsonl)

    return args

class DataPreprocessor: 
    def __init__(self, video_data_path, audio_jsonl, output_path_dir, num_frames=NUM_FRAMES_TO_SAVE_PER_SEGMENT, debug=True):
        
        # prep paths
        self.video_data_path = str(pathlib.Path(video_data_path))
        self.audio_jsonl = str(pathlib.Path(audio_jsonl))
        self.clip_completed_stems_path = str( pathlib.Path(video_data_path).parent / (pathlib.Path(video_data_path).stem + '_clip_completed_stems.jsonl'))
        self.output_path = str(pathlib.Path(output_path_dir))
        
        self.video_file_stems = None
        self.num_frames = num_frames
        self.debug = debug
        
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        print(f"Using {self.device}...")

        self.clip, self.clip_preprocess = clip.load(MODEL_SIZE, self.device)
        if '336' in MODEL_SIZE:
            assert FRAME_SIZE_DIMENSION >= 336, "Frame size must be at least 336px (by 336) for ViT-L/14@336px"
        
        if self.debug:
            print(f"Done setting up CLIP...")

    def get_all_video_file_stems(self):
        ''' 
        Video_file_stems: all input files (not filtered in any way) 
        stem_to_whisper: stem returns list of whisper dicts. 
        ''' 
        # time this function
        start_time = time.monotonic()
        
        if not os.path.exists(self.audio_jsonl):
            print(f"No whisper file exists, can't do CLIP without it. Filepath: {self.audio_jsonl}.")
            exit()
        
        if self.debug:
            print(f"Collecting video file stems in {self.audio_jsonl}")
        self.video_file_stems = [] # all stems
        self.stem_to_whisper = {}  # filename.stem --> whisper_json_object
        with jsonlines.open(self.audio_jsonl) as reader:
            try:
                for _, obj in enumerate(reader):
                    json_objs = json.loads(obj)
                    for json_obj in json_objs:
                        self.video_file_stems.append(json_obj['video_filename_stem'])

                        if json_obj['video_filename_stem'] not in self.stem_to_whisper:
                            self.stem_to_whisper[json_obj['video_filename_stem']] = []
                        
                        self.stem_to_whisper[json_obj['video_filename_stem']].append(json_obj)
            except Exception as e:
                print(f"Error: couldn't read {self.audio_jsonl}. Got error: {e}")
        if self.debug:
            print(f"Done collecting {self.audio_jsonl}")
            
        self.video_file_stems = self.video_file_stems
        self.stem_to_whisper = self.stem_to_whisper
        print(f"⏰ Found {len(self.stem_to_whisper), } whisper objects in {(time.monotonic()-start_time):2f} seconds")
        return self.video_file_stems, self.stem_to_whisper
    
    def get_video_dir_files(self):
        '''
        video_dir_files: set( of all files in video_dir )
        stem_to_filename: {} stem --> filename
        '''
        print("Globbing all video files {}...".format(self.video_data_path))
        start_time = time.monotonic()
        self.video_dir_files = set()
        self.stem_to_filename = {}
        
        for video_dir_file in glob.glob(os.path.join(self.video_data_path, '*'), recursive = True):
            # print(Path(video_dir_file).stem)
            
            # TODO: I think this is not working... Not recognizing all the videos we've done. It's also horribly slow.
            # add a separate json (not jsonl?) that has all the stems we've done. 
            self.video_dir_files.add(str(Path(video_dir_file).stem))
            self.stem_to_filename[str(Path(video_dir_file).stem)] = video_dir_file
        
        print(f"⏰ Globbed {len(self.video_dir_files), } input videos in {(time.monotonic()-start_time):2f} seconds")
        return self.video_dir_files, self.stem_to_filename
    

    def filter_already_completed_video_stems(self,):
        ''' Skil already completed CLIP outputs. '''
        # ensure we collect everything
        if not self.video_file_stems:
            self.get_all_video_file_stems()
        
        if os.path.exists(self.clip_completed_stems_path):
            print(f"Filtering already completed videos in CLIP {self.video_data_path} with existing output in {self.clip_completed_stems_path}")
            already_processed_video_stems = set()
            
            # TODO:
            with jsonlines.open(self.clip_completed_stems_path, mode = 'r') as reader:
                try:
                    for line in reader:
                        if line:
                            already_processed_video_stems.add( json.loads(line) )
                except Exception as e:
                    print(f"Error: couldn't line from {self.output_path}. Got error: {e}")
            print("Already procssed:", already_processed_video_stems)
            remaining_stems_for_clip = set(self.video_file_stems) - set(already_processed_video_stems)
            print(f"Total to process:\t\t\t {len(set(self.video_file_stems))}")
            print(f"Already processed:\t\t\t {len(already_processed_video_stems)}")
            print(f"Remaining to do now:\t\t {len(remaining_stems_for_clip)}")
            self.video_file_stems = list(remaining_stems_for_clip)
            return self.video_file_stems, self.stem_to_whisper
        else:
            # no output file yet, so process everything.
            print("Processing all inputs. No CLIP results yet at filpath: {}".format(self.clip_completed_stems_path))
            return self.video_file_stems, self.stem_to_whisper
        
    def save_video_stem(self, video_stem):
        ''' Save the video stem to the output file. '''
        with jsonlines.open(self.clip_completed_stems_path, mode = 'a') as writer:
            writer.write(json.dumps(video_stem))
    
    def get_frames_for_segments(self, video_filepath: str, segments):
        if len(segments) == 0:
            return None

        # assert os.path.exists(self.stem_to_filename[str(os.path.join(video_name))])
        assert os.path.exists(video_filepath)

        # cap = cv2.VideoCapture(self.stem_to_filename[str(os.path.join(video_name))])
        cap = cv2.VideoCapture(video_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)

        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        segment_frames = []

        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            
            start_frame = int(start_time*fps)
            end_frame = int(end_time*fps)

            curr_segment = []

            if end_frame < start_frame + self.num_frames:
                print(f"[WARNING] Segment has less than {self.num_frames} frames")
                continue
            
            sample_frame_idxs = np.linspace(0, end_frame-start_frame-1, num=self.num_frames, dtype=int)
            for frame_idx in sample_frame_idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, (start_frame+frame_idx)-1)
                # if self.debug:
                #     print(f"reading frame {(start_frame+frame_idx)-1}")
                res, frame = cap.read()
                if res:
                    # if frame exists, resize and append.
                    # curr_segment.append(frame)
                    curr_segment.append(cv2.resize(np.array(frame), dsize=(FRAME_SIZE_DIMENSION, FRAME_SIZE_DIMENSION), interpolation=cv2.INTER_CUBIC))

            segment_frames.append(curr_segment)
        
        return segment_frames

    def get_multimodal_features(self, video_filepath, segments, num_frames=3):
        '''
        Returns 1 whole video worth of info. 1+ segments. 
        
        Return shapes: 
        clip_features    = (segments * num_frames * embd)
        sear_frames      = (segments * num_frames * (frame_input_height * frame_input_width * RGB_CHANNELS))
        caption_features = (segments * embd)
        '''
        if self.debug:
            print("Starting get_multimodal_features")
        segment_frames = self.get_frames_for_segments(video_filepath, segments)

        # scene_graph_features = []
        if self.debug:
            print("Got segment frames")

        image_input = []
        text_inputs = []

        # extract clip features for each segment  and extract text features for captions
        for i, frames_list in enumerate(segment_frames):
            # print(f"Segment {i} has {len(frames_list)} frames")
            if len(frames_list) == 0:
                continue

            assert len(frames_list) <= 3
            
            # WE now resize ALL frames in the get_frames function, for consistent memory usage and everything.
            # resized_frames = [Image.fromarray(cv2.resize(np.array(frames_list[frame_idx]), dsize=(FRAME_SIZE_DIMENSION, FRAME_SIZE_DIMENSION), interpolation=cv2.INTER_CUBIC)) for frame_idx in range(len(frames_list))]
            # segment_frames[i] = resized_frames

            image_input = image_input + [self.clip_preprocess(Image.fromarray(frame)).unsqueeze(0) for frame in segment_frames[i]]
            text_inputs.append(torch.cat([clip.tokenize(segments[i]['caption'])]))
        
        image_input = torch.cat(image_input).to(self.device)
        text_inputs = torch.cat(text_inputs).to(self.device)

        print("RIGHT before running clip 📸")
        start_time = time.monotonic()
        # with torch.no_grad():
        with torch.inference_mode(): # even faster than no_grad()
            image_features = self.clip.encode_image(image_input)
            text_features = self.clip.encode_text(text_inputs)

            image_features = image_features.cpu().numpy().reshape(len(segment_frames), self.num_frames, -1) # -1 == 3.
            text_features = text_features.cpu().numpy()
            # print(f"Time to run clip: {(time.monotonic() - start_time):2f} on {len(segment_frames)* self.num_frames} images")
        
        if self.debug:
            print("Clip features:")
            print(image_features.shape) # only one per segment??
            print("Caption features:")
            print(text_features.shape)
            print("segment_frames:")
            print(np.array(segment_frames).shape)
        
        # all np ndarrays
        return image_features, text_features, np.array(segment_frames)
        

    def run_clip_one_video(self, video_filepath, list_of_whisper_output_dict):
        ''' Basically the main function of this CLIP class. '''
        torch.cuda.empty_cache()
        if self.debug: 
            print("Starting run_clip_one_video")
            print("video_filepath: ", video_filepath)
        
        # initialize empty sample
        whisper_segments = None
        whisper_segments = list_of_whisper_output_dict

        if self.debug:
            print("Loaded Whisper Json file")

        image_features, caption_features, segment_frames = self.get_multimodal_features(video_filepath, whisper_segments, self.num_frames)

        # assert len(image_features) == len(scene_graph_features) == len(caption_features) == len(whisper_segments)

        if self.debug:
            print("Obtained multimodal features")
        
        # Loop over segments. One output per segment. 1+ segment per video.
        list_of_segment_outputs = []
        for i, (image_feature, caption_feature, segment_frame) in enumerate(zip(image_features, caption_features, segment_frames)):
            if self.debug:
                print(f"Formatting segment {i+1} of {len(image_features)}")
            
            # WARNING: Lists need to be wrapped in a dictinoary (see frame_embeddings)
            one_segment_output = {
                "video_stem": str(Path(video_filepath).stem),
                "segment_id": str(Path(video_filepath).stem) + f"_{i}",
                "segment_index": np.int16(i),
                "total_segments": np.int16(len(whisper_segments)),
                "segment_total_time": whisper_segments[i]['end'] - whisper_segments[i]['start'],
                "captions": whisper_segments[i]['caption'],
                "segment_start_time": whisper_segments[i]['start'],
                "segment_end_time": whisper_segments[i]['end'],
                "num_frames_per_segment": np.int16(self.num_frames),
                "frame_embeddings": image_feature,
                "text_caption_embeddings": caption_feature,
                "segment_frames": segment_frame,
                "frame_embeddings_shape": image_feature.shape,          # trying the FLATTEN technique!
                "text_caption_embeddings_shape": caption_feature.shape,
                "segment_frames_shape": segment_frame.shape,
                # "scene_graph_captions": scene_graph_feature -- to be added in the subsequent step.
            }
            # encode & compress each segment, then write them all at once.
            # couldn't do compression cuz json doesn't support bytes. LIKE WTFFF WHY NOT.
            list_of_segment_outputs.append(one_segment_output) # , compression=True, properties={'ndarray_compact': True}
        
        # whole_video_df = xarray.concat(list_of_segment_outputs, dim='video_stem') # index is videostem.
        
        # write all at once. Thread safe. This uses more dram, but might be with slow file IO systems. 
        # Also, writing all at once keeps segments from the same video close together in the json lines, otherwise they're all spread around. 
        if self.debug:
            print(f"Saving to CLIP output path: {self.output_path}")
        
        # SAVE WHOLE VIDEO TO COMPRESSED FILE (named after video_stem)
        np.savez_compressed( Path(self.output_path) / str(Path(video_filepath).stem), *list_of_segment_outputs)
        
        # OLD jsonlines method, too hard to compress.
        # with jsonlines.open(self.output_path, mode='a') as writer:
        #     writer.write_all(list_of_segment_outputs)
        self.save_video_stem(str(Path(video_filepath).stem)) # save completed successfully.
        print("✅ Wrote whole video to jsonlines!!! 😆")

if __name__ == "__main__":
    args = parse_cmd_line_args()
    data_preprocessor = DataPreprocessor(video_data_path=args.video_path, audio_jsonl=args.audio_jsonl, output_path=args.output_path, debug=False)
    # to use this main(), call run_clip_one_video()