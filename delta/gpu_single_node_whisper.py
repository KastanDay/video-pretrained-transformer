''' NOTES for further automation 
ssh <head node>

export CONTAINER=/scratch/bbki/kastanday/whisper/custom_whisper_latest.sif  && \
  singularity run --bind /scratch:/scratch ${CONTAINER} /bin/bash
  
python ~/parallel_pdg/video-pretrained-transformer/delta/gpu_single_node_whisper.py
'''

import os # must be first
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache' # must be first

import sys
# sys.path.append(os.path.join(os.getcwd(),"../data_preprocessing/whisper_audio"))
sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing/whisper_audio")
import CaptionPreprocessing as CaptionPreprocessing

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


def iter_over_input_dirs():
    ''' automation '''
    global FINAL_WHISPER_RESULTS_JSONL
    global REMOTE_VIDEO_DIR
    global LOCAL_VIDEO_DIR
    global LOCAL_RESULTS_JSONL
    global LOCAL_ERRORS_JSONL
    global LOCAL_EMPTY_JSONL
    global NUM_THREADS
    global NUM_CPU_CORES
    global NUM_GPUS
    global GPU_PER_PROCESS
    # THIS is GREAT balance on delta GPU, 4X GPU with clip running
    
    ## TODO: had to ahrdcode these for now
    NUM_THREADS = 68 # first one always dies for some reason.
    NUM_CPU_CORES = 53
    NUM_GPUS = 4
    GPU_PER_PROCESS = 1/18 # 1/25 for 3GPUs. # 1/16 # 1/16 is perfect balance on 4 gpus. Bigger value = more spread across GPUs.
    assert NUM_GPUS/(GPU_PER_PROCESS) >= NUM_THREADS, f"reduce GPU_PER_PROCESS: {GPU_PER_PROCESS} to ensure we have a (portion of a) GPU for each thread."
    
    # get hostname
    result = subprocess.run(["hostname"], capture_output=True, text=True)
    hostname = str(result.stdout.strip())
    
    # 😁 SET ME 😁
    DIRS_TO_PROCESS = [ 
                        'parallel_16',
                        # 'parallel_17',
                        # 'parallel_19',
                        # 'parallel_14',
                    ]
    # Running it twice to pick up any files lost to stochastic reasons...
    for attempts in range(2):
        for dir_name in DIRS_TO_PROCESS:
            if 'hal' in hostname:
                print("ON HAL")
                FINAL_WHISPER_RESULTS_JSONL = f'/home/kastanday/thesis/whisper/{dir_name}_whisper_output.jsonl'
                REMOTE_VIDEO_DIR        = f'/home/kastanday/thesis/whisper/{dir_name}'
                LOCAL_VIDEO_DIR         = f'/tmp/{dir_name}' # used for wavs
                LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
                LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
                LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
            elif any(word in hostname for word in ['gpub', 'gpuc', 'dt-login']):
                print("ON DELTA")
                FINAL_WHISPER_RESULTS_JSONL = f'/scratch/bbki/kastanday/whisper/{dir_name}_whisper_output.jsonl'
                REMOTE_VIDEO_DIR    = f'/scratch/bbki/kastanday/whisper/{dir_name}'
                LOCAL_VIDEO_DIR         = f'/tmp/{dir_name}' # used for wavs
                LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
                LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
                LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
            elif any(word in hostname for word in ['aws', 'ec2']): 
                # TODO
                raise NotImplementedError 
            else:
                raise("No valid hostname error. Exiting")
            
            rsync_inputs_to_workers() # blocking rsync call.
            main()
        # Good vals for Delta CPU nodes. 
        # NUM_THREADS = 3
        # NUM_CPUS = 1
        # GPU_PER_PROCESS = 0 # 1/12 is perfect balance on 4 gpus. Smaller demon = more spread across GPUs.


        # FOR Delta 8x GPU
        # NUM_THREADS = 55*2 # first one always dies for some reason.
        # NUM_CPU_CORES = 58*2
        # NUM_GPUS = 7.5
        # GPU_PER_PROCESS = 1/15 # 1/16 # 1/16 is perfect balance on 4 gpus. Bigger value = more spread across GPUs.


@ray.remote(num_cpus=0.8, num_gpus=1/18) # .70 and 1/30 equals 65% DRAM usage right immediately. Can't really go any higher.
def parallel_caption_extraction(file_batch, itr):
    # todo: get around this this import, but idk why it won't recognize it unless it's in here...
    sys.path.append("/u/kastanday/parallel_pdg/video-pretrained-transformer/data_preprocessing/whisper_audio")
    import CaptionPreprocessing as CaptionPreprocessing
    start = time.monotonic()
    for index, file in enumerate(file_batch):
        process = None
        try:
            # check if output already exists
            out_path = pathlib.Path(file)
            out_path =  pathlib.Path(out_path.with_suffix('.wav')).parts[-1]
            out_path = pathlib.Path(LOCAL_VIDEO_DIR + "_wav" + "/" + out_path)
            
            # todo: check if output exists in whisper file, if so skip, else try again. 
            if os.path.exists(out_path):
                print(f'Input file: f{file} -- already processed, skipping')
            
            # if filename stem in jsonlines. whispers...
            
            # MAIN: run whisper
            process = CaptionPreprocessing.CaptionPreprocessing(FINAL_WHISPER_RESULTS_JSONL)
            process.load_mp4_to_wav_with_outpath(file, out_dir = LOCAL_VIDEO_DIR + "_wav")
            process.get_segments_thresholded()
            process.output_json(LOCAL_VIDEO_DIR)
            
            # todo: os.remove(file) -- this assumes that all files will be copied at the start, and that'll work first try.
            # or save a list of already processed files.
            
            print("✅ Success: ", file)
        except Exception as e:
            # write failed files to jsonlines
            print(f"Error during whisper: {e}")
            failed_file_json_object = json.dumps(str(file))
            error_filepath = LOCAL_VIDEO_DIR + "_whisper_errors.jsonl"
            if not os.path.exists(error_filepath):
                pathlib.Path(error_filepath).touch()
            with jsonlines.open(error_filepath, mode='a') as writer:
                writer.write({"video_filepath": failed_file_json_object, "error": str(e)}) 
        finally:
            # memory savings
            if process:
                del process

        # one file done        
        print(f"⏰ Time to Whisper the file: {(time.monotonic() - start)/60:.2f} minutes\nVideo filesize: {os.path.getsize(file)/1e6:.2f} MB\n")
        start = time.monotonic()

def actual_main():
    
    ''' All this is for distributed ray... only using single node rn.'''
    # init ray
    # result = subprocess.run(["hostname", "-i"], capture_output=True, text=True)
    # head_ip = result.stdout.strip()
    # print(f"Connecting to Ray... at address ray://{head_ip}:10001")
    # ray.init(address=f'ray://{head_ip}:10001', dashboard_port=8265)   # most reliable way to start Ray
    # ray.init(address=f'auto', ignore_reinit_error=True)
    # use port-forwarding to see dashboard: `ssh -L 8265:localhost:8265 kastanday@kingfisher.ncsa.illinois.edu`
    # assert ray.is_initialized() == True
    # print("🎯 Ray initialized.")
    
    # first call, then watch memory usage & restart as necessary
    whisper_thread = threading.Thread(target=iter_over_input_dirs, name="whisper_thread") # , args=some_args
    whisper_thread.start()
    
    while True: 
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        print(f'RAM memory % used: {psutil.virtual_memory()[2]}%')
        if psutil.virtual_memory()[2] > 90: # change to 90 or 95
            print("❌  ⚠️⚠️⚠️Memory usage is high, restarting ray ⚠️⚠️⚠️  ❌")
            ray.shutdown()
            # time.sleep(2)
            assert ray.is_initialized() == False
            
            whisper_thread = threading.Thread(target=iter_over_input_dirs, name="whisper_thread") # , args=some_args
            whisper_thread.start()
            
        time.sleep(120)
        rsync_results_to_scratch()

def main():
    """ MAIN """

    # ray.shutdown()
    ray.init(num_gpus=NUM_GPUS, num_cpus=NUM_CPU_CORES, include_dashboard = False, ignore_reinit_error=True) # , num_gpus = 1
    print_cluster_stats()
    start = time.time()
    
    # glob files in LOCAL_VIDEO_DIR
    print(f"Globbing input files... {REMOTE_VIDEO_DIR}")

    # If we are on the second run, we should get the first case
    if os.path.isfile(FINAL_WHISPER_RESULTS_JSONL):
        file_editor = CaptionPreprocessing.CaptionPreprocessing(FINAL_WHISPER_RESULTS_JSONL)
        files = file_editor.filter_completed_whisper_paths(REMOTE_VIDEO_DIR)
    else:
        files = glob.glob(os.path.join(REMOTE_VIDEO_DIR, '*'), recursive = True)
    
    print(f"Second to glob files: {time.time() - start:.3f}")
    print("Number of files:", len(files))
    
    
    # filter out bad files (.vtt and .wav, and .json) Anything other than webm and mp4?
    files = [ file for file in files if not file.endswith( ('.txt','.vtt', 'json') ) ]
    print("After filtering -- Number of files:", len(files))

    # To check if we complete all files after job
    true_num_files = len(files)
    
    # shuffle the list files
    # random.seed(42) # NO SEED, different shuffle each time.
    random.shuffle(files)
    
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

    if not os.path.isdir(LOCAL_VIDEO_DIR + "_wav"):
        os.mkdir(LOCAL_VIDEO_DIR + "_wav")
        
    # all_results = ray.get([parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)])
    print("Starting parallel batches")
    all_result_futures = [parallel_caption_extraction.remote(batch, itr) for itr, batch in enumerate(batches)]
    
    all_done = ray.get(all_result_futures)

    print("Len of all threads: ", len(all_done))
    print("👉 Completed, finished main().")

def rsync_inputs_to_workers():
    """ Called before processing begins. """
    jsons_process = None
    if os.path.exists(FINAL_WHISPER_RESULTS_JSONL):
        print("Copying jsons to /tmp")
        cp = ['cp', FINAL_WHISPER_RESULTS_JSONL, '/tmp']
        jsons_process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    else:
        print("❌ ERROR: FINAL_RESULTS_DESTINATION does not exist. This is only expected when its your first time processing this file batch.")
        
    
    '''
    👉 video files -- JUST USE STRAIGHT FROM SCRATCH
    SLEEP_TIME = 120
    if we already have some video files locally, don't sleep for long because we already have files ready.
    if os.path.exists(LOCAL_VIDEO_DIR):
        SLEEP_TIME = 30
    if os.path.exists(INPUT_DIR_ON_SCRATCH):
        print("Copying video files to /tmp")
        # cp = ['cp', '-r', INPUT_DIR_ON_SCRATCH, '/tmp']
        cp = [f"rsync", "--update", "-r", INPUT_DIR_ON_SCRATCH, pathlib.Path(LOCAL_VIDEO_DIR).parent]
        process = Popen(cp, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        # don't block for this.
    #   stdout, stderr = process.communicate()
    else:
        print("❌ ERROR: INPUT_DIR_ON_SCRATCH does not exist. Wrong dir specified??")
    '''
    
    # start other copies first, but here we block until we get the curcial jsonL file
    if jsons_process:
        print("important to have the '_jsons' before we start. blocking...")
        stdout, stderr = jsons_process.communicate()
    return

def rsync_results_to_scratch():
    """ Called every minute or so during processing. """
    # non-blocking. Do this frequently.
    
    # FINAL_RESULTS_DESTINATION = f'/scratch/bbki/kastanday/whisper/{dir_name}_output.jsonl'
    # LOCAL_RESULTS_JSONL     = f'/tmp/{dir_name}_whisper_output.jsonl'
    # LOCAL_ERRORS_JSONL      = f'/tmp/{dir_name}_whisper_errors.jsonl'
    # LOCAL_EMPTY_JSONL       = f'/tmp/{dir_name}_whisper_empty.jsonl'
    # LOCAL_CLIP_JSONL       = f'/tmp/{dir_name}_clip_output.jsonl'
    
    whisper_on_scratch = '/scratch/bbki/kastanday/whisper'
    Popen(['rsync', '--update', LOCAL_RESULTS_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    Popen(['rsync', '--update', LOCAL_ERRORS_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    Popen(['rsync', '--update', LOCAL_EMPTY_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # Popen(['rsync', '--update', LOCAL_CLIP_JSONL, whisper_on_scratch], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    # stdout, stderr = process.communicate()
    return

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
    actual_main()