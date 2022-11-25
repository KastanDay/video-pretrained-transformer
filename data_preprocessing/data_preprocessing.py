### Imports
import os 
import json
import cv2
import numpy as np
import clip
import torch
from PIL import Image
from tqdm import tqdm
import pathlib
from pathlib import Path
import glob
import argparse
import jsonlines
import json
import json_numpy
import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq
import fastparquet 
# from fastparquet import ParquetFile, write
import time
from filelock import FileLock

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # just for testing.


# Install depends (STRICT dependencies, mostly due to Ray.):
# conda create -n v3_clip_preprocessing_yt1b python=3.8.13 -y
# conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# pip install pandas "ray[default]==1.13.0" more_itertools jsonlines json_numpy pyarrow fastparquet pandas parquet ftfy regex tqdm git+https://github.com/openai/CLIP.git
# conda install -c conda-forge -y git-lfs
# cd into the git repo and run `git lfs install` and `git lfs pull`
# (optional) pip install pretty_errors

FRAME_SIZE_DIMENSION = 224 

# Best models are (1st) ViT-L/14@336px and (2nd) ViT-L/14. I don't recommend going lower. 
MODEL_SIZE = 'ViT-L/14'

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
    args.output_path = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_clip_output.parquet'))
    args.audio_jsonl = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_whisper_output.jsonl'))

    # validate input
    print("Using video path:", args.video_path)
    assert os.path.exists(args.video_path)
    assert os.path.exists(args.output_path)
    assert os.path.exists(args.audio_jsonl)

    return args

class DataPreprocessor: 
    def __init__(self, video_data_path, audio_jsonl, output_path, num_frames=NUM_FRAMES_TO_SAVE_PER_SEGMENT, debug=True):
        self.video_data_path = video_data_path
        self.audio_jsonl = audio_jsonl
        self.output_path= output_path
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
        return self.video_file_stems, self.stem_to_whisper
    
    def get_video_dir_files(self):
        '''
        video_dir_files: set( of all files in video_dir )
        stem_to_filename: {} stem --> filename
        '''
        print("Globbing all video files {}...".format(self.video_data_path))
        self.video_dir_files = set()
        self.stem_to_filename = {}
        
        for video_dir_file in glob.glob(os.path.join(self.video_data_path, '*'), recursive = True):
            # print(Path(video_dir_file).stem)
            self.video_dir_files.add(str(Path(video_dir_file).stem))
            self.stem_to_filename[str(Path(video_dir_file).stem)] = video_dir_file
        return self.video_dir_files, self.stem_to_filename
    

    def filter_already_completed_video_stems(self,):
        ''' Skil already completed CLIP outputs. '''
        
        print("Filtering already completed videos in CLIP {}...".format(self.video_data_path))
        
        # ensure we collect everything
        if not self.video_file_stems:
            self.get_all_video_file_stems()
        
        # only load 'video_stem' column, for optimal IO.
        if os.path.exists(self.output_path):
            already_processed_video_stems = set(fastparquet.ParquetFile(self.output_path).to_pandas(['video_stem']))
            remaining_stems_for_clip = set(self.video_file_stems) - set(already_processed_video_stems)
            print(f"Total to process:\t\t\t {len(self.video_file_stems)}")
            print(f"Already processed:\t\t\t {len(remaining_stems_for_clip)}")
            print(f"Starting download of remaining:\t\t {len(remaining_stems_for_clip)}")
            self.video_file_stems = list(remaining_stems_for_clip)
            return self.video_file_stems, self.stem_to_whisper
        else:
            # no output file yet, so process everything.
            print("Processing all inputs. No CLIP results yet at filpath: {}".format(self.output_path))
            return self.video_file_stems, self.stem_to_whisper
    
    def get_frames_for_segments(self, video_filepath: str, segments):
        print("Starting get_frames_for_segments")
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
        with torch.no_grad():
            image_features = self.clip.encode_image(image_input)
            text_features = self.clip.encode_text(text_inputs)

            image_features = image_features.cpu().numpy().reshape(len(segment_frames), self.num_frames, -1) # -1 == 3.
            text_features = text_features.cpu().numpy()
            print(f"Time to run clip: {(time.monotonic() - start_time):2f} on {len(segment_frames)* self.num_frames} images")
        
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
        print("Starting run_clip_one_video")
        print("video_filepath: ", video_filepath)
        # print("list_of_whisper_output_dict: ", list_of_whisper_output_dict)
        
        # initialize empty sample
        # whisper_segments = self.stem_to_whisper[video_name] 
        whisper_segments = None
        whisper_segments = list_of_whisper_output_dict

        if self.debug:
            print("Loaded Whisper Json file")

        image_features, caption_features, segment_frames = self.get_multimodal_features(video_filepath, whisper_segments, self.num_frames)

        # assert len(image_features) == len(scene_graph_features) == len(caption_features) == len(whisper_segments)

        if self.debug:
            print("Obtained multimodal features")
        
        list_of_segment_outputs = []
        for i, (image_feature, caption_feature, segment_frame) in enumerate(zip(image_features, caption_features, segment_frames)):
            if self.debug:
                print(f"Formatting segment {i+1} of {len(image_features)}")
            
            # WARNING: Lists need to be wrapped in a dictinoary (see frame_embeddings)
            one_segment_output = pd.DataFrame({
                "video_stem": str(Path(video_filepath).stem),
                "segment_id": str(Path(video_filepath).stem) + f"_{i}",
                "segment_index": i,
                "total_segments": len(whisper_segments),
                "segment_total_time": whisper_segments[i]['end'] - whisper_segments[i]['start'],
                "captions": whisper_segments[i]['caption'],
                "segment_start_time": whisper_segments[i]['start'],
                "segment_end_time": whisper_segments[i]['end'],
                "num_frames_per_segment": self.num_frames,
                "frame_embeddings": image_feature.tobytes(),
                "text_caption_embeddings": caption_feature.tobytes(),
                "segment_frames": segment_frame.tobytes(),
                # "frame_embeddings_shape": image_feature.shape,          # trying the FLATTEN technique!
                # "text_caption_embeddings_shape": caption_feature.shape,
                # "segment_frames_shape": segment_frame.shape,
                # "scene_graph_embeddings": scene_graph_feature -- to be added in the subsequent step.
            }, index=[i])
            # print("one_segment_output pandas df:", one_segment_output) # (135, 768)
            list_of_segment_outputs.append(one_segment_output)
        
        segments_of_entire_video = pd.concat(list_of_segment_outputs, ignore_index=True)
        if self.debug:
            print(segments_of_entire_video)
            print("☝️☝️☝️ ALL SEGMENTS CONCATED pandas right before saving to parquet. Does it have frame embeddings???")
            # single blocking write to parquet.
            # print("saving pandas pickle!")
            # segments_of_entire_video.to_pickle('/scratch/bbki/kastanday/whisper/p14.pickle')
            print(f"Saving to parquet path: {self.output_path}")
        
        lock_file = pathlib.Path(self.output_path).parent / (pathlib.Path(self.output_path).name + '.lock') # same exact filename, with a second .lock extension.
        with FileLock(lock_file): # single threaded writing.
            if os.path.exists(self.output_path):
                print("Appending!! to PARQUET")
                # append if exists
                # ⚠️ ☢️CRUCIAL that stats=False ☢️ (or else memory error chrash: https://github.com/dask/fastparquet/issues/760)
                fastparquet.write(self.output_path, segments_of_entire_video, append=True, compression='snappy', file_scheme='simple', stats=False)
            else:
                print("Writing!! (no append)")
                # try with , 
                fastparquet.write(self.output_path, segments_of_entire_video, append=False, compression='snappy', file_scheme='simple', stats=False)
        if self.debug:
            print("✅ Wrote whole video to parquet file!!! 😆")
            
            # OLD jsonliense method.
            # with jsonlines.open('test_jsonlines.jsonl', mode='a') as writer:
            #     for i, (image_feature, caption_feature, segment_frames) in enumerate(zip(image_features, caption_features, segment_frames)):
            #         # print(f"Processing segment {i+1} of {len(image_features)}")
            #         sample_dict = {
            #             "filename": str(Path(video_filepath).stem),
            #             "segment_length": whisper_segments[i]['end'] - whisper_segments[i]['start'],
            #             "captions": whisper_segments[i]['caption'],
            #             "segment_start_time": whisper_segments[i]['start'],
            #             "segment_end_time": whisper_segments[i]['end'],
            #             "frame_embeddings": image_feature.tobytes(),
            #             "text_caption_embeddings": caption_feature.tobytes(),
            #             "segment_frames": segment_frames.tobytes(),
            #             # "scene_graph_embeddings": scene_graph_feature
            #         }
            #         writer.write(sample_dict) # WRITE output dataset line.

        # if self.debug:
        #     print("Constructed training samples")

            
    # def process_using_audio_dir(self):
    #     samples_not_found = 0
    #     total_samples = 0

    #     # TODO: dataformat
    #     for i in tqdm(range(len(self.video_file_stems))):
    #         video_name = self.video_file_stems[i]
    #         if str(video_name) not in self.video_dir_files:
    #             samples_not_found += 1
    #         else:
    #             if self.debug:
    #                 print("Constructing training samples...")
    #             self.construct_training_samples(video_name, video_filepath)
    #         total_samples += 1

    #     print(f"[❌ ERROR ❌] {samples_not_found} of {total_samples} are invalid")

if __name__ == "__main__":
    args = parse_cmd_line_args()
    data_preprocessor = DataPreprocessor(video_data_path=args.video_path, audio_jsonl=args.audio_jsonl, output_path=args.output_path, debug=False)
    # data_preprocessor.process_using_audio_dir()
    
    