import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first

import sys
sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing") # DELTA
# sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing") # HAL
from data_preprocessing import DataPreprocessor

import pathlib
import time
import json
import ray
import random
import subprocess
from subprocess import PIPE, Popen
import more_itertools
import jsonlines
import traceback

# Good vals for Delta CPU nodes. 
# NUM_THREADS = 3
# NUM_CPUS = 1
# GPU_PER_PROCESS = 0 # 1/12 is perfect balance on 4 gpus. Smaller demon = more spread across GPUs.

# FOR Delta 8x GPU
# NUM_THREADS = 55*2 # first one always dies for some reason.
# NUM_CPU_CORES = 58*2
# NUM_GPUS = 7.5
# GPU_PER_PROCESS = 1/15 # 1/16 # 1/16 is perfect balance on 4 gpus. Bigger value = more spread across GPUs.

def run_main():
    ''' iterate over parallel directories. '''
    
    
    ''' Change directory behavior based on Compute Cluster (Delta / HAL / AWS) '''
    global REMOTE_WHISPER_JSONL_PATH
    global REMOTE_CLIP_OUTPUT_DIR
    global REMOTE_INPUT_VIDEO_PATH
    global LOCAL_INPUT_VIDEO_PATH
    global LOCAL_RESULTS_JSONL
    global LOCAL_ERRORS_JSONL
    global LOCAL_EMPTY_JSONL
    global LOCAL_CLIP_JSONL
    
    global NUM_THREADS
    global NUM_CPU_CORES
    global NUM_GPUS
    global GPU_PER_PROCESS
    
    # THIS is GREAT balance on delta GPU, 4X GPU with clip running
    NUM_THREADS = 16     # Number of parallel processes (limited by DRAM and SRAM)
    NUM_CPU_CORES = 64   # Numer of available physical cores to use (use max!)
    NUM_GPUS = 4         # Number of physical GPUs to use (use max)
    GPU_PER_PROCESS = 1/4 # threads per GPU, limited by OOM errors while also maximizing spread. 
    
    # NUM_CPU_CORES/NUM_THREADS
    
    ''' Setup Ray '''
    result = subprocess.run(["hostname", "-i"], capture_output=True, text=True)
    head_ip = result.stdout.strip()
    print(f"Connecting to Ray... at address ray://{head_ip}:10001")
    # ray.init(address=f'{head_ip}:62158', dashboard_port=8265)   # most reliable way to start Ray
    ray.init(num_cpus=NUM_CPU_CORES, num_gpus=NUM_GPUS, ignore_reinit_error=True)# dashboard_port=8265)
    # ray.init(num_gpus=NUM_CPU_CORES, num_cpus=NUM_THREADS, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
    # use port-forwarding to see dashboard: `ssh -L 8265:localhost:8265 kastanday@kingfisher.ncsa.illinois.edu`
    assert ray.is_initialized() == True
    print("🎯 Ray initialized.")
    print_cluster_stats()
    
    # todo: loop over dirs. but I need to load new ones thru whisper.
    # 😁 SET ME 😁
    DIRS_TO_PROCESS = [ 
                        'parallel_16',
                        # 'parallel_14',
                        # 'parallel_14',
                    ]
    for dir_name in DIRS_TO_PROCESS:
        # get hostname
        result = subprocess.run(["hostname"], capture_output=True, text=True)
        hostname = str(result.stdout.strip())
        
        if 'hal' in hostname:
            REMOTE_WHISPER_JSONL_PATH = f'/home/kastanday/thesis/whisper/{dir_name}_whisper_output.jsonl'
            REMOTE_CLIP_OUTPUT_DIR     = f'/home/kastanday/thesis/whisper/{dir_name}_clip_output'
            REMOTE_INPUT_VIDEO_PATH = f'/home/kastanday/thesis/whisper/{dir_name}'
            LOCAL_INPUT_VIDEO_PATH        = f'/tmp/{dir_name}'
            LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
            LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
            LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
            LOCAL_CLIP_JSONL      = f'/tmp/{dir_name}_clip_output.jsonl'
        elif any(word in hostname for word in ['gpub', 'gpuc', 'dt-login']):
            print("RUNNING ON DELTA")
            REMOTE_WHISPER_JSONL_PATH = f'/scratch/bbki/kastanday/whisper/{dir_name}_whisper_output.jsonl'
            REMOTE_INPUT_VIDEO_PATH = f'/scratch/bbki/kastanday/whisper/{dir_name}'
            REMOTE_CLIP_OUTPUT_DIR  = f'/scratch/bbki/kastanday/whisper/{dir_name}_clip_output'
            LOCAL_INPUT_VIDEO_PATH  = f'/tmp/{dir_name}'
            LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
            LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
            LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
            LOCAL_CLIP_JSONL        = f'/tmp/{dir_name}_clip_output.jsonl'
        elif any(word in hostname for word in ['aws', 'ec2']): # TODO
            raise NotImplementedError 
        else:
            raise("No valid hostname error. Exiting")
        
        # make output dir
        pathlib.Path(REMOTE_CLIP_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        # RUN MAIN with current settings.
        main()

@ray.remote(num_cpus=2, num_gpus=1/4)
def parallel_caption_extraction(file_batch, stem_to_filename, stem_to_whisper):
    start = time.monotonic()
    # sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing")  # Kastan server
    sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing") # Delta
    from data_preprocessing import DataPreprocessor
    my_clip_preprocesser = DataPreprocessor(video_data_path=REMOTE_INPUT_VIDEO_PATH, audio_jsonl=REMOTE_WHISPER_JSONL_PATH, output_path_dir=REMOTE_CLIP_OUTPUT_DIR, debug=False)
    for index, file_stem in enumerate(file_batch):
        try:
            # print(f"Starting file {file_stem}")
            my_clip_preprocesser.run_clip_one_video(stem_to_filename[file_stem], stem_to_whisper[file_stem])# video filepath, whisper_dict_list
            print(f"✅ Time to CLIP the video: ⏰ {(time.monotonic() - start)/60:.2f} minutes\nFile: {file_stem}")
            print(f"")
        except KeyError as ke:
            pass # ignore for testing
            # print("Missing video file: ", file_stem)
            # todo: write to file just like Daniel.
        except Exception as e:
            # write failed files to jsonlines
            print(f"Error during CLIP: {e}\n{traceback.print_exc()}")
            failed_file_json_object = json.dumps(str(file_stem))
            error_filepath = LOCAL_INPUT_VIDEO_PATH + "_whisper_errors.jsonl"
            if not os.path.exists(error_filepath):
                pathlib.Path(error_filepath).touch()
            with jsonlines.open(error_filepath, mode='a') as writer:
                writer.write({"video_filepath": failed_file_json_object, "error": str(e)}) 
        start = time.monotonic()

def main():
    """ NOT the true main(), this is called by run_main() """
    
    sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing")
    from data_preprocessing import DataPreprocessor
    my_clip_preprocesser = DataPreprocessor(video_data_path=REMOTE_INPUT_VIDEO_PATH, audio_jsonl=REMOTE_WHISPER_JSONL_PATH, output_path_dir=REMOTE_CLIP_OUTPUT_DIR, debug=False)
    video_dir_files, stem_to_filename = my_clip_preprocesser.get_video_dir_files()
    video_stems, stem_to_whisper = my_clip_preprocesser.filter_already_completed_video_stems()
    del my_clip_preprocesser # save memory

    # shuffle the list files
    # random.seed(42) # NO SEED, different shuffle each time.
    random.shuffle(video_stems)
    
    # batches = list(more_itertools.divide( int(round(ray.cluster_resources()['CPU'])) , files))
    batches = list(more_itertools.divide( NUM_THREADS , video_stems))
    
    print("Num batches: ", len(batches))
    # print(len(batches), " should equal num threads: ", ray.cluster_resources()['CPU'])
    # assert len(batches) == (ray.cluster_resources()['CPU'])

    # all_results = ray.get([parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)])
    print("Starting parallel batches")
    all_result_futures = [parallel_caption_extraction.remote(batch, stem_to_filename, stem_to_whisper) for itr, batch in enumerate(batches)]
    
    all_done = ray.get(all_result_futures)
    
    print("Len of all threads: ", len(all_done))
    print("👉 Completed, finished main().")

# def rsync_inputs_to_workers():
#     """ Called before processing begins. """
#     jsons_process = None
#     if os.path.exists(FINAL_RESULTS_DESTINATION):
#         print("Copying jsons to /tmp")
#         cp = ['cp', FINAL_RESULTS_DESTINATION, '/tmp']
#         jsons_process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     else:
#         print("❌ ERROR: FINAL_RESULTS_DESTINATION does not exist. This is only expected when its your first time processing this file batch.")
        
#     SLEEP_TIME = 120
#     # if we already have some video files locally, don't sleep for long because we already have files ready.
#     if os.path.exists(INPUT_DIR_TO_TRANSCRIBE):
#         SLEEP_TIME = 30
    
#     # video files
#     if os.path.exists(INPUT_DIR_ON_SCRATCH):
#         print("Copying video files to /tmp")
#         # cp = ['cp', '-r', INPUT_DIR_ON_SCRATCH, '/tmp']
#         cp = [f"rsync", "--update", "-v", INPUT_DIR_ON_SCRATCH, INPUT_DIR_TO_TRANSCRIBE]
#         process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
#         # don't block for this.
#     #   stdout, stderr = process.communicate()
#     else:
#         print("❌ ERROR: INPUT_DIR_ON_SCRATCH does not exist. Wrong dir specified??")
    
#     # start other copies first, but here we block until we get the curcial jsonL file
#     if jsons_process:
#         print("important to have the '_jsons' before we start. blocking...")
#         stdout, stderr = jsons_process.communicate()
    
#     print("Sleeping for %d seconds to donwload the data..." % SLEEP_TIME)
#     time.sleep(SLEEP_TIME)
#     return

# def rsync_results_to_scratch():
#     """ Called every minute or so during processing. """
#     # non-blocking. Do this frequently.
    
#     # FINAL_RESULTS_DESTINATION = f'/scratch/bbki/kastanday/whisper/{dir_name}_output.jsonl'
#     # LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
#     # LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
#     # LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
#     # LOCAL_CLIP_JSONL       = f'/tmp/{dir_name}_clip_output.jsonl'
    
#     whisper_on_scratch = '/scratch/bbki/kastanday/whisper'
#     Popen(['rsync', '--update', LOCAL_RESULTS_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     Popen(['rsync', '--update', LOCAL_ERRORS_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     Popen(['rsync', '--update', LOCAL_EMPTY_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     Popen(['rsync', '--update', LOCAL_CLIP_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     # stdout, stderr = process.communicate()
#     return

def print_cluster_stats():
    print("Querying size of Ray cluster...\n")

    # print at start of staging
    print(f'''This cluster consists of
        {len(ray.nodes())} nodes in total
        {ray.cluster_resources()['CPU']} CPU cores in total
        {ray.cluster_resources()['memory']/1e9:.2f} GB CPU memory in total''')
    if ('GPU' in str(ray.cluster_resources())):
        print(f"        {ray.cluster_resources()['GPU']} GRAPHICCSSZZ cards in total")

if __name__ == '__main__':
    run_main()
    # main()