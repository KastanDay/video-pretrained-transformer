
from logging import raiseExceptions
import os
from os import path
from lhotse import CutSet, RecordingSet, align_with_torchaudio, annotate_with_whisper
# from tqdm import tqdm
from pprint import pprint
from dataclasses import asdict
import torch
from pydub import AudioSegment
import sys

class CaptionPreprocessing:
    # Takes in a path to an mp4 file, converts it to wav
    def __init__(self, debug = False):
        self.debug = debug
        # self.mp3_path = self.load_mp4_to_wav(path)
        self.cut = None

    def process_mp4(self, path):
        self.mp3_path = self.load_mp4_to_wav(path)
        # self.cut = None

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
    def get_segments_thresholded(self, time = 30, threshold = 15):
        def get_cut(path): 
            success = False
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dir = '/'.join(path.split("/")[:-1])
            recordings = RecordingSet.from_dir(dir, pattern="*.wav")
            # Temporary workaround for interval tree error
            index = 0
            while not success:
                index += 1
                # More than 10 failed attempts
                if index > 10:
                    raiseExceptions("Can not annotate this cut")
                try:
                    cuts = annotate_with_whisper(recordings, device = device)
                    cuts_aligned = align_with_torchaudio(cuts)
                    for cut in cuts_aligned:
                        return asdict(cut)
                except ValueError or AssertionError:
                    print("ERROR... restarting")

            
        def to_time_dict():
            time_dict_list = []
            supervision = "ERROR"
            try:
                for supervision in self.cut['supervisions']:
                    # Catch "music" as it doesn't carry semantic meaning, and skip fake text (emojis)

                    if (supervision is None) or (supervision['alignment'] is None) or (supervision['text'] is None) or supervision['text'] == 'Music':
                        print(supervision)
                        continue

                    for word in supervision['alignment']['word']:
                        new_dict = {"word": word.symbol, "start": word.start, "end": word.start + word.duration}
                        time_dict_list.append(new_dict)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                print("CUT", self.cut)
                print("SUPERVISION", supervision)
                return

            return time_dict_list
        if not self.cut:
            try:
                self.cut = get_cut(self.mp3_path)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                return


        time_dict_list = to_time_dict()
        curr_dict_list = []
        index = 0
        while index < (len(time_dict_list)):
            if index + threshold < len(time_dict_list):
                if time_dict_list[index + threshold - 1]["start"] - time_dict_list[index]["end"] <= time:
                    new_dict = {"start": time_dict_list[index]["start"], "end": time_dict_list[index + threshold - 1]["end"],
                    "segment_word_list": time_dict_list[index: index + threshold]}
                    curr_dict_list.append(new_dict)
                    index += threshold
                else:
                    index += 1
            else:
                break
        return curr_dict_list
