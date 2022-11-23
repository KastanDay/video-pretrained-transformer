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
import pyarrow as pa
import pyarrow.parquet as pq
import fastparquet 
from fastparquet import ParquetFile
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # just for testing.


# Install depends (STRICT dependencies, mostly due to Ray.):
# conda create -n v3_clip_preprocessing_yt1b python=3.8.13 -y
# conda install pytorch torchvision torchaudio cudatoolkit=11.7 -c pytorch -c conda-forge -y
# pip install "ray[default]==1.13.0" more_itertools jsonlines json_numpy pyarrow pandas parquet ftfy regex tqdm git+https://github.com/openai/CLIP.git
# conda install -c conda-forge -y git-lfs
# cd into the git repo and run `git lfs install` and `git lfs pull`
# (optional) pip install pretty_errors

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
    def __init__(self, video_data_path, audio_jsonl, output_path, num_frames=3, debug=True):
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

        self.clip, self.clip_preprocess = clip.load('ViT-L/14@336px', self.device)
        if self.debug:
            print(f"Done setting up CLIP...")

    def get_all_video_file_stems(self):
        ''' 
        Video_file_stems: all input files (not filtered in any way) 
        stem_to_whisper: stem returns list of whisper dicts. 
        
        ''' 
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
        self.video_dir_files = set()
        self.stem_to_filename = {}
        
        for video_dir_file in glob.glob(os.path.join(self.video_data_path, '*'), recursive = True):
            # print(Path(video_dir_file).stem)
            self.video_dir_files.add(str(Path(video_dir_file).stem))
            self.stem_to_filename[str(Path(video_dir_file).stem)] = video_dir_file
        return self.video_dir_files, self.stem_to_filename
    

    def filter_already_completed_video_stems(self,):
        ''' Skil already completed CLIP outputs. '''
        # ensure we collect everything
        if not self.video_file_stems:
            self.get_all_video_file_stems()
        
        # only load 'video_stem' column, for optimal IO.
        if os.path.exists(self.output_path):
            already_processed_video_stems = set(ParquetFile(self.output_path).to_pandas(['video_stem']))
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
            if self.debug:
                print("Current segment: ", i)

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

                if self.debug:
                    print(f"reading frame {(start_frame+frame_idx)-1}")

                res, frame = cap.read()

                if self.debug:
                    print("frame read.")

                if res:
                    curr_segment.append(frame)

            segment_frames.append(curr_segment)

        return segment_frames

    def get_multimodal_features(self, video_filepath, segments, num_frames=3):
        segment_frames = self.get_frames_for_segments(video_filepath, segments)
        serialized_frames = []

        # scene_graph_features = []
        clip_features = []
        caption_features = []

        if self.debug:
            print("Got segment frames")

        image_input = []
        text_inputs = []

        # extract clip features for each segment  and extract text features for captions
        for i, frames_list in enumerate(segment_frames):
            print(f"Segment {i} has {len(frames_list)} frames")
            if len(frames_list) == 0:
                continue

            # TODO: extract scene graph features for each segment
            # scene_graph = self.scene_graph_predictor.predict(frames_list[0], num_rel=10)

            serialized_frames.append([json_numpy.dumps(frame) for frame in frames_list])

            # middle_frame_idx = len(frames_list) // 2
            # sample_frame_idxs = [middle_frame_idx]
            # sample_frame_idxs = np.linspace(0, len(frames_list)-1, num=num_frames, dtype=int)

            assert len(frames_list) <= 3
            resized_frames = [Image.fromarray(cv2.resize(np.array(frames_list[frame_idx]), dsize=(224, 224), interpolation=cv2.INTER_CUBIC)) for frame_idx in range(len(frames_list))]

            # image_input = torch.cat([self.clip_preprocess(frame).unsqueeze(0) for frame in resized_frames]).to(self.device)
            # text_inputs = torch.cat([clip.tokenize(segments[i]['caption'])]).to(self.device)

            image_input = image_input + [self.clip_preprocess(frame).unsqueeze(0) for frame in resized_frames]
            text_inputs.append(segments[i]['caption'])
        
        image_input = torch.cat(image_input).to(self.device)
        text_inputs = torch.cat(text_inputs).to(self.device)

        with torch.no_grad():
            image_features = self.clip.encode_image(image_input)
            text_features = self.clip.encode_text(text_inputs)

            image_features = image_features.cpu().numpy().tolist()
            text_features = text_features.cpu().numpy().tolist()



        for i in range(len(segment_frames)):
            num_frames_in_segment = len(segment_frames[i])

            for j in range(num_frames_in_segment):
                clip_features.append(image_features[i][j])

            caption_features.append(text_features[i])


        clip_features.append(image_features.cpu().numpy().tolist())
        caption_features.append(text_features.cpu().numpy().tolist())

        return clip_features, caption_features, serialized_frames
        

    def run_clip_one_video(self, video_filepath, list_of_whisper_output_dict):
        ''' Basically the main function of this CLIP class. '''
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
        for i, (image_feature, caption_feature, segment_frames) in enumerate(zip(image_features, caption_features, segment_frames)):
            print(f"Formatting segment {i+1} of {len(image_features)}")
            # pq.write_table({
            one_segment_output = pd.DataFrame({
                "video_stem": str(Path(video_filepath).stem),
                "segment_id": str(Path(video_filepath).stem) + f"_{i}",
                "segment_index": i,
                "total_segments": len(whisper_segments),
                "segment_total_time": whisper_segments[i]['end'] - whisper_segments[i]['start'],
                "captions": whisper_segments[i]['caption'],
                "segment_start_time": whisper_segments[i]['start'],
                "segment_end_time": whisper_segments[i]['end'],
                "frame_embeddings": image_feature,
                "text_caption_embeddings": caption_feature,
                # "scene_graph_embeddings": scene_graph_feature
                "segment_frames": segment_frames
            }, index=[i])
            
            list_of_segment_outputs.append(one_segment_output)
        
        segments_of_entire_video = pd.concat(list_of_segment_outputs, ignore_index=True)
        # single blocking write to parquet.
        fastparquet.write(self.output_filepath, segments_of_entire_video, append=os.path.exists(self.output_filepath)) # append if exists
        if self.debug:
            print("Constructed training samples")
            
        # todo: remove old jsonlines method
        # with jsonlines.open(self.output_path, mode='a') as writer:
                # sample_dict = {
                #     "video_stem": str(Path(video_filepath).stem),
                #     "segment_id": str(Path(video_filepath).stem) + f"_{i}",
                #     "segment_index": i,
                #     "total_segments": len(whisper_segments),
                #     "segment_total_time": whisper_segments[i]['end'] - whisper_segments[i]['start'],
                #     "captions": whisper_segments[i]['caption'],
                #     "segment_start_time": whisper_segments[i]['start'],
                #     "segment_end_time": whisper_segments[i]['end'],
                #     "frame_embeddings": image_feature,
                #     "text_caption_embeddings": caption_feature,
                #     # "scene_graph_embeddings": scene_graph_feature
                #     "segment_frames": segment_frames
                # }
                # writer.write(sample_dict) # WRITE output dataset line.

            
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
    
    