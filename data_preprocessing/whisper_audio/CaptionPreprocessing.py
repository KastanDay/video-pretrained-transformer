from logging import raiseExceptions
import os
import traceback
import sys
import pathlib
from os import path
import sys
import glob
import json
from dataclasses import asdict

# sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing/whisper_audio/lhotse_faster_whisper/lhotse")
# sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing/whisper_audio/lhotse_faster_whisper")
sys.path.append("/home/kastan/thesis/video-pretrained-transformer/data_preprocessing/whisper_audio/lhotse_faster_whisper/lhotse")
sys.path.append("/home/kastan/thesis/video-pretrained-transformer/data_preprocessing/whisper_audio/lhotse_faster_whisper")
from lhotse import Recording, RecordingSet, align_with_torchaudio
from lhotse import annotator_lhotse

from pydub import AudioSegment
import torch
import jsonlines

class CaptionPreprocessing:
    # Takes in a path to an mp4 file, converts it to wav
    def __init__(self, debug = False):
        # self.final_whisper_results_jsonl = final_whisper_results_jsonl
        self.debug = debug
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.whisper_model = annotator_lhotse("medium", device = self.device)

    def process_mp4(self, path):
        self.cut = None
        self.wav_path = self.load_mp4_to_wav(path)
        # self.cut = None

    def load_mp4_to_wav_with_outpath(self, video_path: str, out_dir: str):
        # Either save in M4A or delete after
        self.video_path = video_path
        self.cut = None
        out_path = pathlib.Path(video_path)
        out_path =  pathlib.Path(out_path.with_suffix('.wav')).parts[-1]
        out_path = pathlib.Path(out_dir + "/" + out_path)
        sound = AudioSegment.from_file(video_path) # format="mp4" (not required)
        sound.export(out_path, format="wav", parameters= ["-ac", "1"], codec='ac3')
        self.wav_path = out_path.as_posix()
        return
    # Converts mp4 to wav
    def load_mp4_to_wav(self, video_path: str):
        """convert my_file.mp4 to my_file.wav"""
        # Might want to make it such that we specify outpath as well
        # command = "ffmpeg -i " + path + " -ab 160k -ac 1 -ar 16000 -vn " + path_to_output_wav
        # subprocess.call(command, shell=True)
        
        out_path = pathlib.Path(video_path)
        out_path = out_path.with_suffix('.wav')
        sound = AudioSegment.from_file(video_path) # format="mp4" (not required)
        sound.export(out_path, format="wav", parameters= ["-ac", "1"])
        return out_path.as_posix()


    def get_segments_thresholded(self, time = 30, threshold = 15):
        """  
        Input: segment time length, threshold of words/segment
        Output: list of dictionaries [{caption: string, start: int, end: int, [{word:, start:, end:,}, ...]}, ...]
        Shape: segment, words in segment, {word: word, start: start, end: end}
        Example output: 
        [
            {'caption': "THERE ARE NO STRANGERS TO LOVE YOU KNOW THE RULES AND SO DO I I'VE",
            'start': 18.9245625, 'end': 27.3031875,
            'segment_word_list': [{'word': 'THERE', 'start': 18.9245625, 'end': 19.0261875}, {'word': 'ARE', 'start': 19.0451875, 'end': 19.106562500000003}, {'word': 'NO', 'start': 19.125625, 'end': 19.307624999999998}, {'word': 'STRANGERS', 'start': 19.3466875, 'end': 19.769875000000003}, {'word': 'TO', 'start': 19.929625, 'end': 20.151812500000002}, {'word': 'LOVE', 'start': 20.3115, 'end': 20.4533125}, {'word': 'YOU', 'start': 22.82325, 'end': 22.965125}, {'word': 'KNOW', 'start': 23.004, 'end': 23.3466875}, {'word': 'THE', 'start': 23.3855, 'end': 23.5073125}, {'word': 'RULES', 'start': 23.606375, 'end': 24.1298125}, {'word': 'AND', 'start': 24.44975, 'end': 24.5515}, {'word': 'SO', 'start': 24.6505625, 'end': 24.912937499999998}, {'word': 'DO', 'start': 25.2128125, 'end': 25.49525}, {'word': 'I', 'start': 25.6545625, 'end': 25.676000000000002}, {'word': "I'VE", 'start': 27.1005, 'end': 27.3031875}]},
        ]
        """
        def get_cut(path): 
            recording = Recording.from_file(path)
            recordings = RecordingSet.from_recordings([recording])
            # Temporary workaround for interval tree error
            # 11/19: There is no work around for this problem without diving into whisper code. We are expecting and handling this error
            index = 0
            while True:
                # More than 10 failed attempts
                try:
                    cuts = self.whisper_model.yield_annotated_recordings(recordings)
                    cuts_aligned = align_with_torchaudio(cuts, device = self.device)
                    for cut in cuts_aligned:
                        return asdict(cut)
                except Exception as e:
                    if index > 4:
                        print("could not get cut:", e)
                        raise
                index += 1

        def to_time_dict():
            time_dict_list = []
            supervision = "ERROR"
            non_english = 0
            try:
                for supervision in self.cut['supervisions']:
                    # Catch "music" as it doesn't carry semantic meaning, and skip fake text (emojis)
                    if supervision['language'] and supervision['language'] != 'en':
                        non_english += 1
                        if non_english > 4:
                            print("Not English... exiting")
                            return []
                        continue
                        
                    if (supervision is None) or (supervision['alignment'] is None) or (supervision['text'] is None) or supervision['text'] == 'Music':
                        continue
                    for word in supervision['alignment']['word']:
                        new_dict = {"word": word.symbol, "start": word.start, "end": word.start + word.duration}
                        time_dict_list.append(new_dict)

            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                raise
            return time_dict_list

        # In case we run get_segments thresholded more than one times
        if self.cut:
            print("We already have the cut!")
        else:
            self.cut = get_cut(self.wav_path)

        time_dict_list = to_time_dict()
        curr_dict_list = []
        index = 0
        segment_index = 0
        while index < (len(time_dict_list)):
            if index + threshold < len(time_dict_list):
                # If the start and end time of n words is within m seconds, add to the list of dictionaries
                # a dictionary with n words
                if time_dict_list[index + threshold - 1]["start"] - time_dict_list[index]["end"] <= time:
                    # Get overarching caption for this time segment
                    caption = " ".join([dic["word"] for dic in time_dict_list[index: index + threshold]])
                    caption = caption.encode('UTF-8', 'ignore')
                    fifteen_word_video_segment_captions = {
                        "caption": caption,
                        "start": float(time_dict_list[index]["start"]),
                        "end": float(time_dict_list[index + threshold - 1]["end"]),
                        "segment_word_list": time_dict_list[index: index + threshold],
                        "video_filename_name": str(pathlib.Path(self.video_path).name),
                        "video_filepath": str(pathlib.Path(self.video_path)),
                        "segment_index": segment_index
                        }
                    segment_index += 1
                    curr_dict_list.append(fifteen_word_video_segment_captions)
                    index += threshold
                else:
                    index += 1
            else:
                break
        # Add total_segments
        num_segments = len(curr_dict_list)
        for segment_dict in curr_dict_list:
          segment_dict["total_segments"] = num_segments
          for word_stamp in segment_dict["segment_word_list"]:
            word_stamp["start"] = float(word_stamp["start"])
            word_stamp["end"] = float(word_stamp["end"])
            word_stamp["word"] = word_stamp["word"].encode(encoding = 'UTF-8', errors = 'ignore')

        # this is the list of words, with timestamps
        self.curr_dict_list = curr_dict_list

        
        # hopefully limit memory usage
        torch.cuda.empty_cache()
        return curr_dict_list

    def output_json(self, input_video_dir):
        if not self.curr_dict_list:
            print("Caption output is empty. Returning...")
            fp = input_video_dir + "_whisper_empty.jsonl"
            with jsonlines.open(fp, mode='a') as writer:
                writer.write(self.video_path)
            return
        json_object = json.dumps(self.curr_dict_list)
        # Parse path name, i.e. kastan/thesis/rick.wav -> rick
        # file_name = "/" + str(pathlib.Path(pathlib.PurePath(self.wav_path).parts[-1]).with_suffix(".json"))
        fp = input_video_dir + "_whisper_output.jsonl"
        with jsonlines.open(fp, mode='a') as writer:
            writer.write(json_object)
        return

# Deprecated ?
    def filter_completed_whisper_paths(self, video_input_dir, empty_file_dir):
        video_input_dir = pathlib.Path(video_input_dir)
        # fp = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_whisper_output.jsonl'))
        fp = str(self.final_whisper_results_jsonl)
        
        # todo: glob all files in input_video_dir
        
        # glob files in INPUT_DIR_TO_TRANSCRIBE
        files = glob.glob(os.path.join(video_input_dir, '*'), recursive = True)
        files = [pathlib.Path(file) for file in files]
        ## EXAMPLE CODE.
        print("Total input video files:", len(files))
        # print(files)
        # whisper_output = str(os.path.join(video_input_dir.parent, video_input_dir.stem + '_whisper_output.jsonl'))

        # self.stem_to_whisper
        # self.output_path

        # self.video_file_stems
        # These are the non-empty files
        existing_whisper_output = set()
        with jsonlines.open(fp, mode = 'r') as reader:
            for line in reader.iter(skip_invalid=True):
                if line:
                    line = json.loads(line)
                    filename = pathlib.Path(line[0]["video_filepath"]).name
                    existing_whisper_output.add(pathlib.Path(os.path.join(video_input_dir, filename)))

        print("👉👉fp", fp)
        fp = str(empty_file_dir)
        # Add empty files that have been processed

        with jsonlines.open(fp, mode = 'r') as reader:
            for line in reader.iter(skip_invalid=True):
                if line:
                    # line = json.loads(line)
                    filename = pathlib.Path(line).name
                    existing_whisper_output.add(pathlib.Path(os.path.join(video_input_dir, filename)))
        # print(existing_whisper_output)

        
        print("Number of existing whisper output:", len(existing_whisper_output))
        remaining_whisper_input = set(files) - set(existing_whisper_output)
        print("Number of remaining whisper input:", len(remaining_whisper_input))
        return list(remaining_whisper_input)

    def get_all_caption_segments(self):
      return self.curr_dict_list