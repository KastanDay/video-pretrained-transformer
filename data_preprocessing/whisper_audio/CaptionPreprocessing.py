
from logging import raiseExceptions
import os
import sys
import pathlib
from os import path
from lhotse import Recording, RecordingSet, align_with_torchaudio
from lhotse import annotator_lhotse
from dataclasses import asdict
from pydub import AudioSegment
import torch
import json
import jsonlines

class CaptionPreprocessing:
    # Takes in a path to an mp4 file, converts it to wav
    def __init__(self, device, debug = False):
        self.debug = debug
        # Model size, device
        self.device = device
        self.whisper_model = annotator_lhotse("base", device = self.device)
        # self.wav_path = self.load_mp4_to_wav(path)

    def process_mp4(self, path):
        self.cut = None
        self.wav_path = self.load_mp4_to_wav(path)
        # self.cut = None

    def load_mp4_to_wav_with_outpath(self, video_path: str, out_dir: str):
        # Have dedicated 
        # Either save in M4A or delete after
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
        
        # todo: more reliable way to get codc
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
            success = False
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            # print("👾 Device:", device)
            # dir = '/'.join(path.split("/")[:-1])
            print(path)

            recording = Recording.from_file(path)
            recordings = RecordingSet.from_recordings([recording])
            # TODO: This is causing problems again. need more help here.
            # Temporary workaround for interval tree error
            index = 0
            while not success:
                index += 1
                # More than 10 failed attempts
                # print("One failed get_cut")
                if index > 10:
                    print("Failed too many times, can't annotate.")
                    raiseExceptions("Can not annotate this cut")
                try:
                    cuts = self.whisper_model.yield_annotated_recordings(recordings)
                    cuts_aligned = align_with_torchaudio(cuts, device = self.device)
                    # print("Successfully got cuts_aligned")
                    for cut in cuts_aligned:
                        return asdict(cut)
                except ValueError or AssertionError as e:
                    print("ERROR... restarting. ", e)

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
                        # print(supervision)
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
                print("Getting cut")
                self.cut = get_cut(self.wav_path)
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print("could not get cut:", exc_type, fname, exc_tb.tb_lineno)
                return
        
        time_dict_list = to_time_dict()
        # print(time_dict_list)
        curr_dict_list = []
        index = 0
        while index < (len(time_dict_list)):
            if index + threshold < len(time_dict_list):
                # print("In while index< leng(timedict)")
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
        # this is the list of words, with timestamps
        self.curr_dict_list = curr_dict_list
        return curr_dict_list

    def output_json(self, dir):
        if not self.curr_dict_list:
            print("Caption output is empty. Returning...")
            return
        # Serializing json
        json_object = json.dumps(self.curr_dict_list)
        
        # Parse path name, i.e. kastan/thesis/rick.wav -> rick
        # file_name = "/" + str(pathlib.Path(pathlib.PurePath(self.wav_path).parts[-1]).with_suffix(".json"))
        fp = dir + "_output.jsonl"
        # print(file_name)
        # with open(dir + file_name, "w") as outfile:
        #     outfile.write(json_object)
        with jsonlines.open(fp, mode='a') as writer:
            writer.write(json_object)
        return
