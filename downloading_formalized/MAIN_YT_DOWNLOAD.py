import ray 
import subprocess
import shlex
import json

NUM_THREADS_IPROYAL = 50
NUM_THREADS_STORM_PROXY = 40

@ray.remote
def iproyal_dl(file_batch, proxy_address):
  """Take in batch of video_ids, download them all. save to same location, and use same download record... should be fine.
  If we restart, they won't re-download, and the threads won't try to download the same video at the same time.
  """
  
  for video_id in file_batch:
    video_id = list(video_id)[0]
    golden_command = f"""yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
    -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_10_thru_49 \
    --download-archive /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/yt_1b_train_download_record_parallel_10_49.txt \
    -P 'temp:/tmp' \
    -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
    --min-views 200 \
    --proxy {proxy_address} \
    --write-subs \
    {video_id}
    """
    
    subprocess.run(shlex.split(golden_command))
  return 

@ray.remote
def stormproxy_dl(file_batch):
  """Take in batch of video_ids, download them all. save to same location, and use same download record... should be fine.
  If we restart, they won't re-download, and the threads won't try to download the same video at the same time.
  
  These two gateways are authorized for 40 threads (set my IP in web gui)
  37.48.118.4:13041
  5.79.66.2:13041
  """
  
  for video_id in file_batch:
    video_id = list(video_id)[0]
    golden_command = f"""yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
    -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_10_thru_49 \
    --download-archive /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/yt_1b_train_download_record_parallel_10_49.txt \
    -P 'temp:/tmp' \
    -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
    --min-views 200 \
    --proxy 37.48.118.4:13041 \
    --write-subs \
    {video_id}
    """
    
    subprocess.run(shlex.split(golden_command))
  return 

def main():
  """ MAIN """
  ray.shutdown()
  ray.init()
  
  YT_TO_DOWNLOAD_ID_LIST = "/home/kastan/thesis/video-pretrained-transformer/downloading_formalized/train_id_collection/tail_train_id_list.txt"
  
  list_of_ids_to_download = []
  import csv
  with open(YT_TO_DOWNLOAD_ID_LIST) as csvfile:
    list_of_ids_to_download = list(csv.reader(csvfile))
    
  # make a batch for each download thread (so threads don't try to download the same file)
  batches = make_batch(list_of_ids_to_download, len(list_of_ids_to_download)//(NUM_THREADS_IPROYAL + NUM_THREADS_STORM_PROXY - 1))
  print("Batch size: ", len(batches[0]))
  print("Num batches: ", len(batches))
  
  print(len(batches), " should equal num threads: ", NUM_THREADS_IPROYAL + NUM_THREADS_STORM_PROXY)
  assert len(batches) == (NUM_THREADS_IPROYAL + NUM_THREADS_STORM_PROXY)
  
  # Launch parallel 
  assert len(new_iproyal_proxies) == NUM_THREADS_IPROYAL
  futures =      [iproyal_dl.remote(batches[i], proxy_address) for i, proxy_address in enumerate(new_iproyal_proxies)]
  
  # use batches in index 50 to 89.
  futures.extend([stormproxy_dl.remote(batches[i]) for i in range(NUM_THREADS_IPROYAL, NUM_THREADS_STORM_PROXY)])

  # Retrieve results.
  all_results = ray.get(futures)

  print(len(all_results))
  print(all_results)
  print("👉 Completed, finished main().")
  

def make_batch(items, batch_size):
    """
    Simple helper.
    Create batches of a given size from a list of items.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
  
new_iproyal_proxies = [
  'socks5://14a49cc959431:a403d3be86@74.117.114.28:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.10:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.16:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.238:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.171:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.8:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.251:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.152:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.27:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.18:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.145:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.155:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.7:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.185:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.11:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.128:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.213:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.131:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.240:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.144:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.250:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.149:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.141:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.187:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.175:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.5:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.174:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.89:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.96:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.177:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.102:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.102:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.130:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.10:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.123:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.176:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.86:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.19:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.230:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.185:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.169:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.161:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.207:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.166:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.12:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.115.255:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.44:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.93:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.208:12324',
  'socks5://14a49cc959431:a403d3be86@74.117.114.94:12324',
]
  
if __name__ == '__main__':
  main()
