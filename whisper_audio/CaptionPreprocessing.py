
import subprocess
import os
# Imports 
from lhotse import CutSet, RecordingSet, align_with_torchaudio, annotate_with_whisper
from tqdm import tqdm
from pprint import pprint
from dataclasses import asdict
import torch
from os import path
from pydub import AudioSegment

class CaptionPreprocessing:
    # Takes in a path to an mp4 file, converts it to wav
    def __init__(self, debug = False):
        self.debug = debug
        # self.mp3_path = self.load_mp4_to_wav(path)
        # self.cut = None

    def process_mp4(self, path):
        self.mp3_path = self.load_mp4_to_wav(path)
        self.cut = None

    # Converts mp4 to wav
    def load_mp4_to_wav(self, path):
        # convert my_file.mp4 to my_file.wav
        # base_name =  os.path.splitext(path)[0]
        # path_to_output_wav = base_name + '.wav'
        # command = "ffmpeg -i " + path + " -ab 160k -ac 1 -ar 16000 -vn " + path_to_output_wav
        # subprocess.call(command, shell=True)
        # print("sucess")
        base_name =  os.path.splitext(path)[0]
        dst = base_name + '.wav'


        # convert mp4 to wav
        sound = AudioSegment.from_file(path,format="mp4")
        sound.export(dst, format="wav", parameters= ["-ac", "1"])
        return dst



    # Input: segment time length, threshold of words/segment
    # Output: list of dictionary {start:, end:, [{word:, start:, end:,}, ...]} 
    # shape: segment, words in segment, {word: word, start: start, end: end}
    def get_segments_thresholded(self, cut, time = 30, threshold = 15):
        def get_cut(path): 
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dir = '/'.join(path.split("/")[:-1])
            recordings = RecordingSet.from_dir(dir, pattern="*.wav")
            cuts = annotate_with_whisper(recordings, device = device)
            cuts_aligned = align_with_torchaudio(cuts)
            for cut in cuts_aligned:
                return asdict(cut)
            
        def to_time_dict():
            time_dict_list = []
            for supervision in self.cut['supervisions']:
                if supervision['text'] == 'Music':
                    continue
                for word in supervision['alignment']['word']:
                    new_dict = {"word": word.symbol, "start": word.start, "end": word.start + word.duration}
                    time_dict_list.append(new_dict)
            return time_dict_list
        # if not self.cut:
        #     self.cut = get_cut(self.mp3_path)
        self.cut = cut
        time_dict_list = to_time_dict()
        curr_dict_list = []
        index = 0
        while index < (len(time_dict_list)):
            if index + threshold < len(time_dict_list):
                if time_dict_list[index + threshold]["start"] - time_dict_list[index]["end"] <= time:
                    new_dict = {"start": time_dict_list[index]["start"], "end": time_dict_list[index + threshold + 1]["end"],
                    "segment_word_list": time_dict_list[index: index + threshold]}
                    curr_dict_list.append(new_dict)
                    index += threshold
                else:
                    index += 1
            else:
                break
        return curr_dict_list
