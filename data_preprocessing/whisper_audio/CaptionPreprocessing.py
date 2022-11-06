
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
        
        # todo: more reliable way to get codc
        # p = Path(path_to_json)
        # p = p.with_suffix('.npy')
        
        base_name =  os.path.splitext(path)[0]
        dst = base_name + '.wav'


        # convert mp4 to wav
        sound = AudioSegment.from_file(path,format="mp4") # todo: get codec from filename
        sound.export(dst, format="wav", parameters= ["-ac", "1"])
        return dst



    # Input: segment time length, threshold of words/segment
    # Output: list of dictionaries [{caption: string, start: int, end: int, [{word:, start:, end:,}, ...]}, ...]
    # shape: segment, words in segment, {word: word, start: start, end: end}
    # Example output: [{'caption': "THERE ARE NO STRANGERS TO LOVE YOU KNOW THE RULES AND SO DO I I'VE",
    #  'start': 18.9245625, 'end': 27.3031875,
    #  'segment_word_list': [{'word': 'THERE', 'start': 18.9245625, 'end': 19.0261875}, {'word': 'ARE', 'start': 19.0451875, 'end': 19.106562500000003}, {'word': 'NO', 'start': 19.125625, 'end': 19.307624999999998}, {'word': 'STRANGERS', 'start': 19.3466875, 'end': 19.769875000000003}, {'word': 'TO', 'start': 19.929625, 'end': 20.151812500000002}, {'word': 'LOVE', 'start': 20.3115, 'end': 20.4533125}, {'word': 'YOU', 'start': 22.82325, 'end': 22.965125}, {'word': 'KNOW', 'start': 23.004, 'end': 23.3466875}, {'word': 'THE', 'start': 23.3855, 'end': 23.5073125}, {'word': 'RULES', 'start': 23.606375, 'end': 24.1298125}, {'word': 'AND', 'start': 24.44975, 'end': 24.5515}, {'word': 'SO', 'start': 24.6505625, 'end': 24.912937499999998}, {'word': 'DO', 'start': 25.2128125, 'end': 25.49525}, {'word': 'I', 'start': 25.6545625, 'end': 25.676000000000002}, {'word': "I'VE", 'start': 27.1005, 'end': 27.3031875}]},
    # ...]

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
                # print(exc_type, fname, exc_tb.tb_lineno)
                # print("CUT", self.cut)
                # print("SUPERVISION", supervision)
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
                # If the start and end time of n words is within m seconds, add to the list of dictionaries
                # a dictionary with n words
                if time_dict_list[index + threshold - 1]["start"] - time_dict_list[index]["end"] <= time:
                    # Get overarching caption for this time segment
                    caption = " ".join([dic["word"] for dic in time_dict_list[index: index + threshold]])
                    new_dict = {"caption": caption,"start": time_dict_list[index]["start"], "end": time_dict_list[index + threshold - 1]["end"],
                    "segment_word_list": time_dict_list[index: index + threshold]}
                    curr_dict_list.append(new_dict)
                    index += threshold
                else:
                    index += 1
            else:
                break
        return curr_dict_list
