import os
import tqdm
import time
import traceback
import numpy as np
import glob
import cv2
import pathlib
import deeplake
from termcolor import colored
# import jsonlines
# import json

BATCH_NAME          = 'parallel_15'
DATASET_PATH        = f'/mnt/storage_ssd/FULL_whisper_results_{BATCH_NAME}'
BASE_DIR            = f'/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/'
REMOTE_WHISPER_FILE = f'{BASE_DIR}/{BATCH_NAME}_whisper_output.jsonl'
REMOTE_CLIP_DIR     = f'{BASE_DIR}/{BATCH_NAME}_clip_output'
REMOTE_SCENE_FILE   = f'{BASE_DIR}/{BATCH_NAME}_scene_output.jsonl'

def main():
    ''' Create dataset, if not exists '''
    print(colored(f"Warning, creating fresh dataset every time", "yellow", attrs=["reverse", "bold"]))
    if False and os.path.exists(DATASET_PATH):
        # raise NotImplementedError
        ds = deeplake.load(DATASET_PATH)
        print(ds.summary())
        all_stems = ds.video_stem.data()['value']
        already_completed_stems = set(all_stems)
        print("already completed: ", len(already_completed_stems))
    else:
        print("Creating new dataset, sleeping 3 seconds to allow you to cancel.")  
        time.sleep(3)
        ds = deeplake.empty(DATASET_PATH, overwrite=True)
        print(colored(f"New dataset created at path: {DATASET_PATH}", "cyan", attrs=["reverse", "bold"]))
        already_completed_stems = set()  # empty
        with ds:
            # these fileds are created by Whisper (and in parallel_whisper.py)
            ds.create_tensor('caption',        htype='text', dtype=str, sample_compression=None)
            ds.create_tensor('video_filename', htype='text', dtype=str, sample_compression=None)
            ds.create_tensor('video_filepath', htype='text', dtype=str, sample_compression=None)
            ds.create_tensor('segment_metadata', htype='text', dtype=str, sample_compression=None)
            # out_ds.create_tensor('frames',         htype='image', dtype=np.uint8, sample_compression='jpeg')
            # out_ds.create_tensor('caption_embedding', htype='image', dtype=np.float16, sample_compression='lz4')
            # out_ds.create_tensor('clip_last_hidden_states', htype='image', dtype=np.float32, sample_compression='lz4')
            # out_ds.create_tensor('clip_pooled_embedding', htype='image', dtype=np.float32, sample_compression='lz4')
            # out_ds.create_tensor('timestamp', htype='generic', dtype=float, sample_compression='lz4')
            
            
            # segment_frames          = ds.create_tensor('segment_frames', htype='image', dtype=np.uint8, sample_compression='jpg') # compression
            # video_name              = ds.create_tensor('video_name', htype='text', dtype=str, sample_compression=None)
            # text_caption_embeddings = ds.create_tensor('text_caption_embeddings', htype='image', dtype=np.float16, sample_compression='lz4') # CLIP produces FP16 embeddings.
            # frame_embeddings        = ds.create_tensor('frame_embeddings', htype='image', dtype=np.float16, sample_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html
            # segment_metadata_json   = ds.create_tensor('segment_metadata', htype='json', sample_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html

    print(f"Globbing paths from {REMOTE_CLIP_DIR}")
    clip_paths = glob.glob(os.path.join(REMOTE_CLIP_DIR, '*'), recursive = True)
    novel_filepaths = []
    print("Loading clip files")
    for path in tqdm.tqdm(clip_paths):
        if pathlib.Path(path).stem not in already_completed_stems:
            novel_filepaths.append(path)
    print("Adding these stems now:", len(novel_filepaths))
    print("\n\t".join(novel_filepaths[:10]))
    
    # go from stems to full filepaths (For repeating CLIP)
    stem_to_filepath_dict = create_stem_to_filepath_dict()
        
    # assert find parallel_15_clip_completed | wc -l == clip_path_and_scene_graph + len(all_stems)
    # import more_itertools 
    # batches_of_inputs = more_itertools.chunked(novel_filepaths, 100)
    # for batch in batches_of_inputs:
    try:
        with ds:    
            file_to_deeplake(stem_to_filepath_dict=stem_to_filepath_dict).eval(novel_filepaths, ds, scheduler='ray', num_workers=11)
    except Exception as e:
        print(e)

    print("✅ Full file done.")
    print(ds.summary())


@deeplake.compute
def file_to_deeplake(clip_npz_path, sample_out, stem_to_filepath_dict):
    try:
        clip_npz_path = str(pathlib.Path(clip_npz_path))
        np_loaded = np.load(clip_npz_path, allow_pickle=True)
        # scene_seg_list = json.loads(scene_seg)
    except Exception as e:
        print(f"Failed to load compressed numpy {clip_npz_path}\n{e}")
        
        return -1
    
    # iterate over segments
    for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
        try:
            video_stem = np_loaded[f'arr_{segment_index}'].item()["video_stem"]
            segment_index_val = np_loaded[f'arr_{segment_index}'].item()["segment_index"]
            total_segments = np_loaded[f'arr_{segment_index}'].item()["total_segments"]
            caption = np_loaded[f'arr_{segment_index}'].item()["captions"]
            segment_start_time = np_loaded[f'arr_{segment_index}'].item()["segment_start_time"]
            segment_end_time = np_loaded[f'arr_{segment_index}'].item()["segment_end_time"]
            # frame_embedding       = np_loaded[f'arr_{segment_index}'].item()['frame_embeddings']
            # caption_embedding     = np_loaded[f'arr_{segment_index}'].item()['text_caption_embeddings']
            # segment_id = np_loaded[f'arr_{segment_index}'].item()["segment_id"]
            # segment_total_time = np_loaded[f'arr_{segment_index}'].item()["segment_total_time"]
            # num_frames_per_segment = np_loaded[f'arr_{segment_index}'].item()["num_frames_per_segment"]
            # frame_embeddings = np_loaded[f'arr_{segment_index}'].item()["frame_embeddings"]
            # text_caption_embeddings = np_loaded[f'arr_{segment_index}'].item()["text_caption_embeddings"]
            # segment_frames = np_loaded[f'arr_{segment_index}'].item()["segment_frames"]
            # frame_embeddings_shape = np_loaded[f'arr_{segment_index}'].item()["frame_embeddings_shape"]
            # text_caption_embeddings_shape = np_loaded[f'arr_{segment_index}'].item()["text_caption_embeddings_shape"]
            # segment_frames_shape = np_loaded[f'arr_{segment_index}'].item()["segment_frames_shape"]
            
            # print(f"type of scene seg list {type(scene_seg_list)}")
            # print(f"total_segments vs len(scene_seg_list): {total_segments} vs {len(scene_seg_list)}")
            
            # convert color before saving segment_frames
            # segment_frames = np.uint8(segment_frames.reshape(336, 336, 3))
            # segment_frames = cv2.cvtColor(segment_frames, cv2.COLOR_BGR2RGB)
            # segment_frames = np.asarray(segment_frames, dtype="uint8" )

            local_segment_metadata = {
                'segment_index': int(segment_index_val),
                'total_segments': int(total_segments),
                'captions': caption,
                'start': float(segment_start_time),
                'end': float(segment_end_time),
                # 'video_stem': video_stem,
                # 'segment_id': segment_id,
                # 'segment_total_time': float(segment_total_time),
                # 'num_frames_per_segment': int(num_frames_per_segment),
                # 'segment_frames_shape': segment_frames_shape,
                # 'text_caption_embeddings_shape': text_caption_embeddings_shape,
                # 'frame_embeddings_shape': frame_embeddings_shape,
                # 'scene_graph_caption_txt': scene_seg_list[int(segment_index_val)]
                # 'segment_frames': ,
                # 'text_caption_embeddings': ,
                # 'frame_embeddings': ,
            }
            
            sample_out.segment_metadata.append(local_segment_metadata)
            sample_out.caption.append(caption)
            sample_out.video_filepath.append(stem_to_filepath_dict[video_stem])
            sample_out.video_filename.append(pathlib.Path(stem_to_filepath_dict[video_stem]).name)
            
        except Exception as e:
            print(f"ERROR while adding to sample_out, on file: {clip_npz_path}")
            # print(f"Scene graph list: {scene_seg}. Len: {len(scene_seg)}")
            # print(f"Scene graph index value: {segment_index_val}")
            traceback.print_exc()
            return -1
    # finish whole video of data.
    return sample_out

def create_stem_to_filepath_dict():
    '''
    glob all files
    create dict of stem -> filepath
    '''
    vids_dir = os.path.join(BASE_DIR, BATCH_NAME)
    print(f"Globbing paths from {vids_dir}")
    vid_paths = glob.glob(os.path.join(vids_dir, '*'), recursive=True)
    stem_to_filepath_dict = {}
    for path in tqdm.tqdm(vid_paths):
        stem_to_filepath_dict[pathlib.Path(path).stem] = path
    return stem_to_filepath_dict
    
    

if __name__ == "__main__":
    main()
# print("WandB Logging (full dataset upload???)")
# Docs: https://docs.deeplake.ai/en/latest/Weights-and-Biases.html
# import wandb
# run = wandb.init(project="deeplake_wandb", job_type="dataset_upload")
# ds.commit("creation") # commit -> trigger logging
# run.finish()