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

def parse_cmd_line_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_path', type=str, help="path of video directory")
    parser.add_argument('--output_path', type=str, help="path of output directory")
    parser.add_argument('--audio_jsonl', type=str, help="path to json lines files containing audios")

    args = parser.parse_args()
    print("args: ", args)
    
    return args


class DataPreprocessor: 
    def __init__(self, video_data_path, audio_jsonl, num_frames=3, debug=True):
        self.video_data_path = video_data_path
        self.audio_jsonl = audio_jsonl
        self.audio_data_path = str(self.video_data_path).replace(Path(self.video_data_path).name, Path(self.video_data_path).name + "_json")

        self.num_frames = num_frames
        self.debug = debug

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.debug:
            print(f"Using {self.device}...")

        self.clip, self.clip_preprocess = clip.load('ViT-B/32', self.device)

        if self.debug:
            print(f"Done setting up CLIP...")

        # self.scene_graph_predictor = Predictor()
        # self.scene_graph_predictor.setup()

        self.video_dir_files = set()
        self.stem_to_filename = {}
        
        for video_dir_file in glob.glob(os.path.join(self.video_data_path, '*'), recursive = True):
            # print(Path(video_dir_file).stem)
            self.video_dir_files.add(str(Path(video_dir_file).stem))
            self.stem_to_filename[str(Path(video_dir_file).stem)] = video_dir_file



    # def get_frames_for_segments(self, video_name, segments):
        # if len(segments) == 0:
        #     return None
        
        # curr_segment_idx = 0
        # curr_segment_start = segments[0]['start']
        # curr_segment_end = segments[0]['end']

        # # segment_frames = [[] for i in range(len(segments))]
        # segment_frames = []
        # used_frames = [[] for i in range(len(segments))]

        # # assert os.path.exists(self.video_data_path+video_name+extension)
        # assert os.path.exists(self.stem_to_filename[str(os.path.join(video_name))])

        # # cap = cv2.VideoCapture(self.video_data_path+video_name+extension)
        # cap = cv2.VideoCapture(self.stem_to_filename[str(os.path.join(video_name))])
        # fps = cap.get(cv2.CAP_PROP_FPS)

        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
        
        # size = (frame_width, frame_height)

        # timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
        # curr_timestamp = 0.0

        # segments_done = False
        
    #     if self.debug:
    #         print(f"Getting frames for {len(segments)} segments...")

    #     while(cap.isOpened()):
    #         frame_exists, curr_frame = cap.read()
    #         # print(f"Getting frame {curr_timestamp}...")
            
    #         if frame_exists:
    #             curr_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    #             timestamps.append(curr_timestamp)

    #             if (curr_timestamp/1000.0) < curr_segment_start:
    #                 continue

    #             else:
    #                 if (curr_timestamp/1000.0) < curr_segment_end:
    #                     # segment_frames[curr_segment_idx].append(curr_frame)
    #                     segment_frames.append(curr_frame)
    #                 else:
    #                     curr_segment_start = segments[curr_segment_idx]['start']
    #                     curr_segment_end = segments[curr_segment_idx]['end']

    #                     sample_frame_idxs = np.linspace(0, len(segment_frames)-1, num=self.num_frames, dtype=int)
    #                     used_frames[curr_segment_idx] = [segment_frames[frame_idx].tolist() for frame_idx in sample_frame_idxs]

    #                     while True:
    #                         if ((curr_timestamp/1000.0) > curr_segment_start) and ((curr_timestamp/1000.0) < curr_segment_end):
    #                             break

    #                         if (curr_timestamp/1000.0) < curr_segment_start:
    #                             break

    #                         curr_segment_idx += 1
                            
    #                         if self.debug:
    #                             print("Current segment: ", curr_segment_idx, " with length: ", segments[curr_segment_idx]['end']- segments[curr_segment_idx]['start'])

    #                         segment_frames = []
    #                         if curr_segment_idx >= len(segments):
    #                             # print("segments done hit")
    #                             segments_done = True
    #                             break
                        
    #                         curr_segment_start = segments[curr_segment_idx]['start']
    #                         curr_segment_end = segments[curr_segment_idx]['end']

    #                     if segments_done:
    #                         break
                
    #         else:
    #             break
        

    #     # segment_frames[curr_segment_idx] = np.linspace(0, len(frames_list)-1, num=num_frames, dtype=int)
        



    #     cap.release()
    #     # print(min(timestamps), max(timestamps))

    #     return used_frames

    def get_frames_for_segments(self, video_name, segments):
        if len(segments) == 0:
            return None

        # assert os.path.exists(self.video_data_path+video_name+extension)
        assert os.path.exists(self.stem_to_filename[str(os.path.join(video_name))])

        # cap = cv2.VideoCapture(self.video_data_path+video_name+extension)
        cap = cv2.VideoCapture(self.stem_to_filename[str(os.path.join(video_name))])
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


    def get_multimodal_features(self, video_name, segments, num_frames=3):
        segment_frames = self.get_frames_for_segments(video_name, segments)
        serialized_frames = []

        # scene_graph_features = []
        clip_features = []
        caption_features = []

        if self.debug:
            print("Got segment frames")


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

            image_input = torch.cat([self.clip_preprocess(frame).unsqueeze(0) for frame in resized_frames]).to(self.device)
            text_inputs = torch.cat([clip.tokenize(segments[i]['caption'])]).to(self.device)
        
            with torch.no_grad():
                image_features = self.clip.encode_image(image_input)
                text_features = self.clip.encode_text(text_inputs)

                clip_features.append(image_features.cpu().numpy().tolist())
                caption_features.append(text_features.cpu().numpy().tolist())




        return clip_features, caption_features, serialized_frames


    def construct_training_samples(self, video_name, output_path):
        # initialize empty sample
        whisper_segments = None
        with open(self.audio_data_path+video_name+".json", "r") as whisper_f:
            whisper_segments = json.load(whisper_f)

        if self.debug:
            print("Loaded Whisper Json file")

        image_features, caption_features, segment_frames = self.get_multimodal_features(video_name, whisper_segments, self.num_frames)

        # assert len(image_features) == len(scene_graph_features) == len(caption_features) == len(whisper_segments)

        if self.debug:
            print("Obtained multimodal features")

        
        for i, (image_feature, caption_feature, segment_frames) in enumerate(zip(image_features, caption_features, segment_frames)):
            print(f"Processing segment {i+1} of {len(image_features)}")
            sample_dict = {
                "filename": video_name,
                "segment_length": whisper_segments[i]['end'] - whisper_segments[i]['start'],
                "captions": whisper_segments[i]['caption'],
                "segment_start_time": whisper_segments[i]['start'],
                "segment_end_time": whisper_segments[i]['end'],
                "frame_embeddings": image_feature,
                "text_caption_embeddings": caption_feature,
                # "scene_graph_embeddings": scene_graph_feature
                "segment_frames": segment_frames
            }

            # TODO: Uncomment this code to start saving the files
            # np.save(os.path.join(output_path, f"{video_name}_segment{i}"), sample_dict)
            with jsonlines.open(os.path.join(output_path, f"dataset.jsonl"), mode='a') as writer:
                writer.write(sample_dict)

        if self.debug:
            print("Constructed training samples")
            
    def process_using_audio_dir(self, output_path):
        samples_not_found = 0
        total_samples = 0

        if not os.path.exists(output_path):
            # Create a new directory because it does not exist
            os.makedirs(output_path)
        
        # with jsonlines.open(self.audio_jsonl) as reader:
        #     for obj_idx, obj in enumerate(reader):
        #         print(obj)
        #         1/0

        all_whisper_files = os.listdir(self.audio_data_path)
        for i in tqdm(range(len(all_whisper_files))):
            f = all_whisper_files[i]

            suffix = Path(os.path.join(self.audio_data_path, f)).suffix
            if suffix == ".json":
                video_name = Path(os.path.join(self.audio_data_path, f)).stem
                # todo: use os.path.join()

                if str(video_name) not in self.video_dir_files:
                    # print(video_name)
                    samples_not_found += 1
                else:

                    if self.debug:
                        print("Constructing training samples...")

                    self.construct_training_samples(video_name, output_path)

                total_samples += 1
        
        if self.debug:
            print(f"[WARNING] {samples_not_found}/{total_samples} are invalid")
                


if __name__ == "__main__":
    args = parse_cmd_line_args()

    # webm, mp4, avi, mkv
    data_preprocessor = DataPreprocessor(video_data_path=args.video_path, audio_jsonl=args.audio_jsonl, debug=True)
    data_preprocessor.process_using_audio_dir(args.output_path)
    