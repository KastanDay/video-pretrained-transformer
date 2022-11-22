import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first

import sys
# sys.path.append(os.path.join(os.getcwd(),"../data_preprocessing/whisper_audio"))
sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing")
# import CaptionPreprocessing as CaptionPreprocessing
from data_preprocessing import DataPreprocessor
# print(sys.path)

import pathlib
import time
import json
import ray
import random
import glob
import subprocess
from subprocess import PIPE, Popen
import psutil
import shlex
import more_itertools
import threading
import jsonlines
import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

global FINAL_RESULTS_DESTINATION
global INPUT_DIR_ON_SCRATCH
global INPUT_DIR_TO_TRANSCRIBE
global LOCAL_RESULTS_JSONL
global LOCAL_ERRORS_JSONL
global LOCAL_EMPTY_JSONL
global LOCAL_CLIP_JSONL
# get hostname
result = subprocess.run(["hostname"], capture_output=True, text=True)
hostname = str(result.stdout.strip())

dir_name = 'parallel_12' # 😁 SET ME 😁
if 'hal' in hostname:
    REMOTE_WHISPER_JSONL_PATH = f'/home/kastanday/thesis/whisper/{dir_name}_whisper_output.jsonl'
    REMOTE_CLIP_JSONL_PATH = f'/home/kastanday/thesis/whisper/{dir_name}_clip_output.jsonl'
    REMOTE_INPUT_VIDEO_PATH = f'/home/kastanday/thesis/whisper/{dir_name}'
    LOCAL_VIDEO_PATH = f'/tmp/{dir_name}'
    LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
    LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
    LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
    LOCAL_CLIP_JSONL        = f'/tmp/{dir_name}_clip_output.jsonl'
elif any(word in hostname for word in ['gpub', 'gpuc', 'dt-login']):
    FINAL_RESULTS_DESTINATION = f'/scratch/bbki/kastanday/whisper/{dir_name}_whisper_output.jsonl'
    INPUT_DIR_ON_SCRATCH = f'/scratch/bbki/kastanday/whisper/{dir_name}'
    INPUT_DIR_TO_TRANSCRIBE = f'/tmp/{dir_name}'
    LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
    LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
    LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
    LOCAL_CLIP_JSONL        = f'/tmp/{dir_name}_clip_output.jsonl'
elif any(word in hostname for word in ['aws', 'ec2']): # TODO
    raise NotImplementedError 
else:
    raise("No valid hostname error. Exiting")

# Good vals for Delta CPU nodes. 
# NUM_THREADS = 3
# NUM_CPUS = 1
# GPU_PER_PROCESS = 0 # 1/12 is perfect balance on 4 gpus. Smaller demon = more spread across GPUs.

# THIS is GREAT balance on delta GPU, 4X GPU with clip running
# NUM_THREADS = 55 # first one always dies for some reason.
# NUM_CPU_CORES = 58
# NUM_GPUS = 3.7
# GPU_PER_PROCESS = 1/16 # 1/16 # 1/16 is perfect balance on 4 gpus. Bigger value = more spread across GPUs.

# FOR Delta 8x GPU
# NUM_THREADS = 55*2 # first one always dies for some reason.
# NUM_CPU_CORES = 58*2
# NUM_GPUS = 7.5
# GPU_PER_PROCESS = 1/15 # 1/16 # 1/16 is perfect balance on 4 gpus. Bigger value = more spread across GPUs.

# FOR HAL 
GPU_PER_PROCESS = 1/10 #lots of CPUs per GPU.


# assert NUM_GPUS/(GPU_PER_PROCESS) >= NUM_THREADS

@ray.remote(num_cpus=0.8, num_gpus=GPU_PER_PROCESS) # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
def parallel_caption_extraction(file_batch, itr):
    # todo: get around this this import, but idk why it won't recognize it unless it's in here...
    # sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing")
    # from data_preprocessing import DataPreprocessor
    start = time.monotonic()
    sys.path.append("/home/kastanday/thesis/video-pretrained-transformer/data_preprocessing")
    from data_preprocessing import DataPreprocessor
    data_preprocessor = DataPreprocessor(video_data_path=REMOTE_INPUT_VIDEO_PATH, audio_jsonl=REMOTE_WHISPER_JSONL_PATH, output_path=REMOTE_CLIP_JSONL_PATH, debug=False)
    for index, file in enumerate(file_batch):
        try:
            data_preprocessor.construct_training_samples(file)
            
            # todo: better resuming. if filename stem in jsonlines.... 
            print("✅ Success: ", file)
        except Exception as e:
            # write failed files to jsonlines
            print(f"Error during whisper: {e}")
            failed_file_json_object = json.dumps(str(file))
            error_filepath = INPUT_DIR_TO_TRANSCRIBE + "_whisper_errors.jsonl"
            if not os.path.exists(error_filepath):
                pathlib.Path(error_filepath).touch()
            with jsonlines.open(error_filepath, mode='a') as writer:
                writer.write({"video_filepath": failed_file_json_object, "error": str(e)}) 

        # one file done        
        print(f"⏰ Time to Whisper the file: {(time.monotonic() - start)/60:.2f} minutes")
        start = time.monotonic()

# def run_main():
#     # rsync_inputs_to_workers() # blocking rsync call.
    
#     # first call, then watch memory usage & restart as necessary
#     whisper_thread = threading.Thread(target=main, name="whisper_thread") # , args=some_args
#     whisper_thread.start()
    
#     while True: 
#         print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
#         print(f'RAM memory % used: {psutil.virtual_memory()[2]}%')
#         if psutil.virtual_memory()[2] > 90: # change to 90 or 95
#             print("❌  ⚠️⚠️⚠️Memory usage is high, restarting ray ⚠️⚠️⚠️  ❌")
#             ray.shutdown()
#             # time.sleep(2)
#             assert ray.is_initialized() == False
            
#             whisper_thread = threading.Thread(target=main, name="whisper_thread") # , args=some_args
#             whisper_thread.start()
            
#         time.sleep(120)
#         rsync_results_to_scratch()

def main():
    """ MAIN """
    # init ray
    result = subprocess.run(["hostname", "-i"], capture_output=True, text=True)
    head_ip = result.stdout.strip()
    print(f"Connecting to Ray... at address ray://{head_ip}:10001")
    # ray.init(address=f'ray://{head_ip}:10001', dashboard_port=8265)   # most reliable way to start Ray
    ray.init(address=f'auto', ignore_reinit_error=True, dashboard_port=8265)
    # use port-forwarding to see dashboard: `ssh -L 8265:localhost:8265 kastanday@kingfisher.ncsa.illinois.edu`
    print(f"Port forward with command:\n\t\tssh -L 8265:localhost:8265")
    assert ray.is_initialized() == True
    print("🎯 Ray initialized.")

    # ray.init(num_gpus=NUM_CPU_CORES, num_cpus=NUM_THREADS, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
    print_cluster_stats()
    start = time.time()
    
    sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing")
    from data_preprocessing import DataPreprocessor
    my_clip_preprocesser = DataPreprocessor(video_data_path=REMOTE_INPUT_VIDEO_PATH, audio_jsonl=REMOTE_WHISPER_JSONL_PATH, output_path=REMOTE_CLIP_JSONL_PATH, debug=True)
    audio_file_stems = my_clip_preprocesser.get_audio_file_stems()
    video_dir_files = my_clip_preprocesser.get_video_dir_files()

    files = []
    samples_not_found = 0
    total_samples = 0
    print("Loading video filenames")
    for video_name in audio_file_stems:
        if str(video_name) not in video_dir_files:
            samples_not_found += 1
        else:
            files.append(video_name)
            # self.construct_training_samples(video_name)
        total_samples += 1
    
    if samples_not_found > 0:
      print(f"[❌ WARNING ❌] {samples_not_found} of {total_samples} are invalid")

    # shuffle the list files
    # random.seed(42) # NO SEED, different shuffle each time.
    random.shuffle(files)
    
    # batches = list(more_itertools.divide( int(round(ray.cluster_resources()['CPU'])) , files))
    batches = list(more_itertools.divide( 10 , files))
    
    # print batch stats
    batchsize_iterator = 0
    for i, val in enumerate(batches[0]):
        if i < 5:
            print(val)
        batchsize_iterator = i
    print("Batch size: ", batchsize_iterator)
    print("Num batches: ", len(batches))
    # print(len(batches), " should equal num threads: ", ray.cluster_resources()['CPU'])
    # assert len(batches) == (ray.cluster_resources()['CPU'])

    # all_results = ray.get([parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)])
    print("Starting parallel batches")
    all_result_futures = [parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)]
    
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
    # run_main()
    main()