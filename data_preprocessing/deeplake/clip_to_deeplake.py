import deeplake
import jsonlines
import tqdm
import numpy as np
import json
import glob
import traceback
import os
import time
import cv2
import pathlib

dataset_name        = '/mnt/storage_ssd/FULL_parallel_ingest_p17'
BASE_DIR            = '/mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/'
# BASE_DIR            = '/mnt/storage_ssd/'
BATCH_NAME          = 'parallel_17'
# BASE_DIR            = '/scratch/bbki/kastanday/whisper'
# MODEL_SAVE_PATH     = f'{BASE_DIR}/MODEL_CHECKPOINTS/{MODEL_VERSION_NAME}'
REMOTE_WHISPER_FILE = f'{BASE_DIR}/{BATCH_NAME}_whisper_output.jsonl'
REMOTE_CLIP_DIR     = f'{BASE_DIR}/{BATCH_NAME}_clip_output'
REMOTE_SCENE_FILE   = f'{BASE_DIR}/{BATCH_NAME}_scene_output.jsonl'

def main():
    ''' Create dataset '''
    if os.path.exists(dataset_name):
        ds = deeplake.load(dataset_name)
        print(ds.summary())
        all_stems = ds.video_stem.data()['value']
        already_completed_stems = set(all_stems)
        print("already completed: ", len(already_completed_stems))
    else:
        print("Creating new dataset, sleeping 6 seconds to allow you to cancel.")  
        time.sleep(6)
        ds = deeplake.empty(dataset_name, overwrite=True)
        already_completed_stems = set()  # empty
        with ds:
            segment_frames          = ds.create_tensor('segment_frames', htype='image', dtype=np.uint8, sample_compression='jpg')
            video_stem              = ds.create_tensor('video_stem', htype='text', dtype=str, sample_compression=None)
            text_caption_embeddings = ds.create_tensor('text_caption_embeddings', htype='image', dtype=np.float16, sample_compression='lz4') # CLIP produces FP16 embeddings.
            frame_embeddings        = ds.create_tensor('frame_embeddings', htype='image', dtype=np.float16, sample_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html
            segment_metadata_json   = ds.create_tensor('segment_metadata', htype='json', sample_compression='lz4') # compression: https://docs.deeplake.ai/en/latest/Compressions.html

    print(f"Globbing paths from {REMOTE_CLIP_DIR}")
    clip_paths = glob.glob(os.path.join(REMOTE_CLIP_DIR, '*'), recursive = True)
    novel_filepaths = []
    print("Loading clip files")
    for path in tqdm.tqdm(clip_paths):
        if pathlib.Path(path).stem not in already_completed_stems:
            novel_filepaths.append(path)
    print("Adding these stems now:", len(novel_filepaths))
    print("\n".join(novel_filepaths[:10]))
        
    # assert find parallel_15_clip_completed | wc -l == clip_path_and_scene_graph + len(all_stems)


    import more_itertools 
    batches_of_inputs = more_itertools.chunked(novel_filepaths, 100)
    for batch in batches_of_inputs:
        try:
            with ds:    
                file_to_deeplake().eval(batch, ds, scheduler='ray', num_workers=11)
        except Exception as e:
            print(e)


    print("✅ Full file done.")
    print(ds.summary())


@deeplake.compute
def file_to_deeplake(clip_npz_path, sample_out):
    try:
        clip_npz_path = str(pathlib.Path(clip_npz_path))
        np_loaded = np.load(clip_npz_path, allow_pickle=True)
        # scene_seg_list = json.loads(scene_seg)
    except Exception as e:
        print(f"Failed to load compressed numpy {clip_npz_path}\n{e}")
        
        return -1
    
    video_stem = np_loaded['arr_0'].item()["video_stem"]
    
    # iterate over segments
    for segment_index in range(np_loaded['arr_0'].item()['total_segments']):
        try:
            # frame_embedding       = np_loaded[f'arr_{segment_index}'].item()['frame_embeddings']
            # caption_embedding     = np_loaded[f'arr_{segment_index}'].item()['text_caption_embeddings']
            video_stem = np_loaded[f'arr_{segment_index}'].item()["video_stem"]
            segment_id = np_loaded[f'arr_{segment_index}'].item()["segment_id"]
            segment_index_val = np_loaded[f'arr_{segment_index}'].item()["segment_index"]
            total_segments = np_loaded[f'arr_{segment_index}'].item()["total_segments"]
            segment_total_time = np_loaded[f'arr_{segment_index}'].item()["segment_total_time"]
            captions = np_loaded[f'arr_{segment_index}'].item()["captions"]
            segment_start_time = np_loaded[f'arr_{segment_index}'].item()["segment_start_time"]
            segment_end_time = np_loaded[f'arr_{segment_index}'].item()["segment_end_time"]
            num_frames_per_segment = np_loaded[f'arr_{segment_index}'].item()["num_frames_per_segment"]
            frame_embeddings = np_loaded[f'arr_{segment_index}'].item()["frame_embeddings"]
            text_caption_embeddings = np_loaded[f'arr_{segment_index}'].item()["text_caption_embeddings"]
            segment_frames = np_loaded[f'arr_{segment_index}'].item()["segment_frames"]
            frame_embeddings_shape = np_loaded[f'arr_{segment_index}'].item()["frame_embeddings_shape"]
            text_caption_embeddings_shape = np_loaded[f'arr_{segment_index}'].item()["text_caption_embeddings_shape"]
            segment_frames_shape = np_loaded[f'arr_{segment_index}'].item()["segment_frames_shape"]
            
            # print(f"type of scene seg list {type(scene_seg_list)}")
            # print(f"total_segments vs len(scene_seg_list): {total_segments} vs {len(scene_seg_list)}")
            
            # convert color before saving segment_frames
            segment_frames = np.uint8(segment_frames.reshape(336, 336, 3))
            segment_frames = cv2.cvtColor(segment_frames, cv2.COLOR_BGR2RGB)
            segment_frames = np.asarray(segment_frames, dtype="uint8" )

            
            local_segment_metadata = {
                'video_stem': video_stem,
                'segment_id': segment_id,
                'segment_index': int(segment_index_val),
                'total_segments': int(total_segments),
                'captions': captions,
                'segment_total_time': float(segment_total_time),
                'segment_start_time': float(segment_start_time),
                'segment_end_time': float(segment_end_time),
                'num_frames_per_segment': int(num_frames_per_segment),
                'segment_frames_shape': segment_frames_shape,
                'text_caption_embeddings_shape': text_caption_embeddings_shape,
                'frame_embeddings_shape': frame_embeddings_shape,
                # 'scene_graph_caption_txt': scene_seg_list[int(segment_index_val)]
                # 'segment_frames': ,
                # 'text_caption_embeddings': ,
                # 'frame_embeddings': ,
            }
            
            # add to dataset, WITHIN our group (video_stem)
            # print("Segment_frames.size", segment_frames.size, "should equal", 336*336*3)
            assert 336 * 336 * 3 == segment_frames.size, print("Segment_frames.size", segment_frames.size, "should equal", 336*336*3)  # warning image dimension is hard coded 
            sample_out.segment_frames.append(segment_frames) # warning image dimension is hard coded
            sample_out.text_caption_embeddings.append(np.float16(text_caption_embeddings))
            sample_out.frame_embeddings.append(np.float16(frame_embeddings))
            sample_out.video_stem.append(str(video_stem))
            sample_out.segment_metadata.append(local_segment_metadata)
            # print(f"Success on segment {segment_index} of {total_segments}")
        except Exception as e:
            print(f"ERROR while adding to sample_out, on file: {clip_npz_path}")
            # print(f"Scene graph list: {scene_seg}. Len: {len(scene_seg)}")
            # print(f"Scene graph index value: {segment_index_val}")
            traceback.print_exc()
            return -1
    # finish whole video of data.
    return sample_out


if __name__ == "__main__":
    main()
# print("WandB Logging (full dataset upload???)")
# Docs: https://docs.deeplake.ai/en/latest/Weights-and-Biases.html
# import wandb
# run = wandb.init(project="deeplake_wandb", job_type="dataset_upload")
# ds.commit("creation") # commit -> trigger logging
# run.finish()