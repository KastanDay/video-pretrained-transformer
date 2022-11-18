import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first
import CaptionPreprocessing as CaptionPreprocessing

import sys
sys.path.append("./lhotse_holder/lhotse")

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


# have (just 3-5) more threads than cores so that if one finishes early, it we will stil saturate the CPU longer...
# dir_name = 'parallel_10_thru_49_part_2'
dir_name = 'parallel_49'
# dir_name = 'handpicked_downloads'
FINAL_RESULTS_DESTINATION = f'/scratch/bbki/kastanday/whisper/{dir_name}_output.jsonl'
INPUT_DIR_ON_SCRATCH = f'/scratch/bbki/kastanday/whisper/{dir_name}'
INPUT_DIR_TO_TRANSCRIBE = f'/tmp/{dir_name}'
LOCAL_RESULTS_JSONL = f'/tmp/{dir_name}_output.jsonl'

# TODO: Set huggingface cache to /tmp for faster model loading.

# Good vals for Delta CPU nodes. 
NUM_THREADS = 3
NUM_CPUS = 1
GPU_PER_PROCESS = 0 # 1/12 is perfect balance on 4 gpus. Smaller demon = more spread across GPUs.

# Good vals for Delta GPU nodes. 
# NUM_THREADS = 90
# NUM_CPUS = 1
# GPU_PER_PROCESS = 1/16 # 1/12 is perfect balance on 4 gpus. Smaller demon = more spread across GPUs.


# , num_gpus = GPU_PER_PROCESS
@ray.remote(num_cpus = NUM_CPUS) # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
def parallel_caption_extraction(file_batch, itr):
    print("In parallel_caption_extraction.")
    

    # Write these to disk
    failed_files = []
    # start = time.time()
    for index, file in enumerate(file_batch):
        process = None
        try:
            # check if output already exists
            out_path = pathlib.Path(file)
            out_path =  pathlib.Path(out_path.with_suffix('.wav')).parts[-1]
            out_path = pathlib.Path(INPUT_DIR_TO_TRANSCRIBE + "_wav" + "/" + out_path)
            if os.path.exists(out_path):
                print(f'Input file: f{file} -- already processed, skipping')
                continue
            
            # MAIN: run whisper
            print("Starting caption preprocessing")
            process = CaptionPreprocessing.CaptionPreprocessing()
            process.load_mp4_to_wav_with_outpath(file, out_dir = INPUT_DIR_TO_TRANSCRIBE + "_wav")
            process.get_segments_thresholded()
            process.output_json(INPUT_DIR_TO_TRANSCRIBE)
            print("✅ Success: ", file)
        except Exception as e:
            print(f"Error during whisper: {e}")
            # We could do this if we don't want to count files that failed
            failed_files.append(str(file))
        finally:
            # memory savings
            if process:
                del process

        # one file done        
        # print(f"⏰ Time to Whisper the file: {(time.time() - start)/60:.2f} minutes\nVideo filesize: {os.path.getsize(file)/1e6:.2f} MB\n")
        # start = time.time()
            
    # TODO: this line was erroring out.
    # TODO: "TypeError: Object of type PosixPath is not JSON serializable"
    # todo: all append to the same file.
    try:
        if len(failed_files) > 0:
          with open(f'/tmp/whisper_failed_files_{itr}.json', 'w') as f:
              json.dump(failed_files, f, indent=2)
    except Exception as e:
        print(f"Error writing failed files list: {e}")

def run_main():
    # init ray
    # result = subprocess.run(["hostname", "-i"], capture_output=True, text=True)
    # head_ip = result.stdout.strip()
    # print(f"Connecting to Ray... at address ray://{head_ip}:10001")
    # ray.init(address=f'ray://{head_ip}:10001', dashboard_port=8265)   # most reliable way to start Ray
    # ray.init(address=f'auto', ignore_reinit_error=True)
    # use port-forwarding to see dashboard: `ssh -L 8265:localhost:8265 kastanday@kingfisher.ncsa.illinois.edu`
    # assert ray.is_initialized() == True
    # print("🎯 Ray initialized.")
    
    # TODO: renable this...
    rsync_inputs_to_workers() # blocking rsync call.
    
    # first call, then watch memory usage & restart as necessary
    whisper_thread = threading.Thread(target=main, name="whisper_thread") # , args=some_args
    whisper_thread.start()
    
    while True: 
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        print(f'RAM memory % used: {psutil.virtual_memory()[2]}%')
        if psutil.virtual_memory()[2] > 90: # change to 90 or 95
            print("❌  ⚠️⚠️⚠️Memory usage is high, restarting ray ⚠️⚠️⚠️  ❌")
            ray.shutdown()
            # time.sleep(2)
            assert ray.is_initialized() == False
            
            whisper_thread = threading.Thread(target=main, name="whisper_thread") # , args=some_args
            whisper_thread.start()
            
        time.sleep(60)
        rsync_results_to_scratch()

def main():
    """ MAIN """
    # ray.shutdown()
    ray.init(num_gpus = 0, num_cpus = 2, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
    print_cluster_stats()
    start = time.time()
    
    # glob files in INPUT_DIR_TO_TRANSCRIBE
    print(f"Globbing input files... {INPUT_DIR_TO_TRANSCRIBE}")
    files = glob.glob(os.path.join(INPUT_DIR_TO_TRANSCRIBE, '*'), recursive = True)
    print(f"Second to glob files: {time.time() - start:.3f}")
    print("Number of files:", len(files))
    
    # todo: filter out bad files (.vtt and .wav, and .json) Anything other than webm and mp4?
    
    files = [ file for file in files if not file.endswith( ('.txt','.vtt', 'json') ) ]
    print("After filtering -- Number of files:", len(files))
    
    # shuffle the list files
    # random.seed(42) # NO SEED, different shuffle each time.
    random.shuffle(files)
    
    print("MUST REMOVE THIS LINE destroyes all files. dddddddddddddddddddddd")
    files = files[:3]
    
    if NUM_THREADS == 1:
        batches = [files]
    else:
        batches = list(more_itertools.divide(NUM_THREADS, files))
    
    # print batch stats
    batchsize_iterator = 0
    for i, val in enumerate(batches[0]):
        batchsize_iterator = i
    print("Batch size: ", batchsize_iterator)
    print("Num batches: ", len(batches))
    print(len(batches), " should equal num threads: ", NUM_THREADS)
    assert len(batches) == (NUM_THREADS)

    if not os.path.isdir(INPUT_DIR_TO_TRANSCRIBE + "_wav"):
        os.mkdir(INPUT_DIR_TO_TRANSCRIBE + "_wav")
        
    # all_results = ray.get([parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)])
    print("Starting parallel batches")
    all_result_futures = [parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)]
    
    all_done = ray.get(all_result_futures)
    
    # for i in range(0, len(all_result_futures)): 
    #     # ray.wait(app_futures) catches ONE at a time. 
    #     ready, not_ready = ray.wait(all_result_futures) # todo , fetch_local=False do not download object from remote nodes
    #     print(f"👋 One thread done. Completed {i+1} of {len(batches)}, {(i+1)/len(batches)*100:.2f}%, ⏰ Elapsed time: {(time.time() - start)/60:.2f} min.\n🔮 Estimated total runtime: { ((time.time() - start)/60) / ((i+1)/len(batches)) :.2f} minutes.\n")
    #     all_result_futures = not_ready
    #     if not all_result_futures:
    #         break

    print(all_result_futures)
    print("Len of all threads: ", len(all_result_futures))
    print("👉 Completed, finished main().")

def rsync_inputs_to_workers():
    """ Called before processing begins. """
    jsons_process = None
    if os.path.exists(FINAL_RESULTS_DESTINATION):
        print("Copying jsons to /tmp")
        cp = ['cp', FINAL_RESULTS_DESTINATION, '/tmp']
        jsons_process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    else:
        print("❌ ERROR: FINAL_RESULTS_DESTINATION does not exist. This is only expected when its your first time processing this file batch.")
        
    SLEEP_TIME = 120
    # if we already have some video files, don't sleep for 2 min, just go. 
    if os.path.exists(INPUT_DIR_ON_SCRATCH):
        SLEEP_TIME = 0
    
    # video files
    if os.path.exists(INPUT_DIR_ON_SCRATCH):
        print("Copying video files to /tmp")
        cp = ['cp', '-r', INPUT_DIR_ON_SCRATCH, '/tmp']
        process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # don't block for this.
    #   stdout, stderr = process.communicate()
    else:
        print("❌ ERROR: INPUT_DIR_ON_SCRATCH does not exist. Wrong dir specified??")
    
    # start other copies first, but here we block until we get the curcial jsonL file
    if jsons_process:
        print("important to have the '_jsons' before we start. blocking...")
        stdout, stderr = jsons_process.communicate()
    
    print("Sleeping for %d seconds to donwload the data..." % SLEEP_TIME)
    time.sleep(SLEEP_TIME)
    return

def rsync_results_to_scratch():
    """ Called every minute during processing. """
    # jsons
    rsync = ['rsync', '--update', LOCAL_RESULTS_JSONL, FINAL_RESULTS_DESTINATION]
    process = Popen(rsync, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # non-blocking. Do this frequently.
    # stdout, stderr = process.communicate()

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