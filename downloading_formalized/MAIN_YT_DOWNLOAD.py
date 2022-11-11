import ray 
import subprocess
import shlex
import more_itertools

NUM_THREADS_IPROYAL = 55
NUM_THREADS_STORM_RESIDENTIAL = 1
NUM_THREADS_RAW_NO_PROXY = 10
TOTAL_THREADS = NUM_THREADS_IPROYAL + NUM_THREADS_STORM_RESIDENTIAL + NUM_THREADS_RAW_NO_PROXY

YT_TO_DOWNLOAD_ID_LIST = "/home/kastan/thesis/video-pretrained-transformer/downloading_formalized/train_id_collection/LONG_tail_train_id_list.txt"
DOWNLOAD_ARCHIVE = "/home/kastan/thesis/video-pretrained-transformer/downloading_formalized/train_id_collection/yt_1b_train_download_record_parallel_10_49.txt"

@ray.remote(num_cpus = 0.01)
def iproyal_dl(file_batch, proxy_address):
  """Take in batch of video_ids, download them all. save to same location, and use same download record... should be fine.
  If we restart, they won't re-download, and the threads won't try to download the same video at the same time.
  """
  for video_id in file_batch:
    golden_command = f"""yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
    -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_10_thru_49 \
    --download-archive {DOWNLOAD_ARCHIVE} \
    -P 'temp:/tmp' \
    -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
    --min-views 200 \
    --proxy {proxy_address} \
    --write-subs \
    -N 10 \
    "{video_id}"
    """
    subprocess.run(shlex.split(golden_command))
  return 

@ray.remote(num_cpus = 0.01)
def stormproxy_residential_dl(file_batch):
  """
  Stormproxy residential port
  69.30.217.114:19014
  """
  for video_id in file_batch:
    golden_command = f"""yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
    -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_10_thru_49 \
    --download-archive {DOWNLOAD_ARCHIVE} \
    -P 'temp:/tmp' \
    -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
    --min-views 200 \
    --proxy 69.30.217.114:19014 \
    --write-subs \
    -N 5 \
    "{video_id}"
    """
    subprocess.run(shlex.split(golden_command))
  return 

@ray.remote(num_cpus = 0.01)
def raw_no_proxy_dl(file_batch):
  """
  RAW connection, no proxy. Why not? 
  """
  for video_id in file_batch:
    golden_command = f"""yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
    -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_10_thru_49 \
    --download-archive {DOWNLOAD_ARCHIVE} \
    -P 'temp:/tmp' \
    -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
    --min-views 200 \
    --write-subs \
    -N 5 \
    "{video_id}"
    """
    subprocess.run(shlex.split(golden_command))
  return 

def make_download_list():
  # Filter already downloaded vids:  
  import csv
  with open(YT_TO_DOWNLOAD_ID_LIST) as csvfile:
    list_of_ids_to_download = list(csv.reader(csvfile))
  list_of_ids_to_download = [val[0] for val in list_of_ids_to_download if len(val) > 0]
  del list_of_ids_to_download[0] # delete header

  with open(DOWNLOAD_ARCHIVE) as csvfile:
    already_downloaded = list(csv.reader(csvfile))
  already_downloaded = [val[0].replace('youtube ', '') for val in already_downloaded]

  remaining_to_download = set(list_of_ids_to_download) - set(already_downloaded)
  print(f"Total to download:\t\t\t {len(list_of_ids_to_download)}")
  print(f"Already downloaded:\t\t\t {len(already_downloaded)}")
  print(f"Starting download of remaining:\t\t {len(remaining_to_download)}")
  return list(remaining_to_download)

def main():
  """ MAIN """
  ray.shutdown()
  ray.init(include_dashboard = False)
  
  list_of_ids_to_download = make_download_list()
  print(list_of_ids_to_download[:20])
  batches = list(more_itertools.divide(TOTAL_THREADS, list_of_ids_to_download))
  
  # print batch stats
  counter = 0
  for i, val in enumerate(batches[0]):
    counter = i
  print("Batch size: ", counter)
  print("Num batches: ", len(batches))
  print(len(batches), " should equal num threads: ", TOTAL_THREADS)
  assert len(batches) == (TOTAL_THREADS)

  # Launch parallel 
  assert len(new_iproyal_proxies) == NUM_THREADS_IPROYAL
  futures =      [iproyal_dl.remote(batches[i], proxy_address) for i, proxy_address in enumerate(new_iproyal_proxies)]
  # next set, use batches in index 50 to 89.
  # futures.extend([stormproxy_dl.remote(batches[i]) for i in range(NUM_THREADS_IPROYAL, NUM_THREADS_IPROYAL+NUM_THREADS_STORM_PROXY)])
  futures.extend([stormproxy_residential_dl.remote(batches[i]) for i in range(NUM_THREADS_IPROYAL, NUM_THREADS_IPROYAL+NUM_THREADS_STORM_RESIDENTIAL)])
  futures.extend([raw_no_proxy_dl.remote(batches[i]) for i in range(NUM_THREADS_IPROYAL+NUM_THREADS_STORM_RESIDENTIAL, TOTAL_THREADS)])
  
  # make sure we launched all the jobs
  assert len(futures) == TOTAL_THREADS

  # Retrieve results.
  all_results = ray.get(futures)

  print(len(all_results))
  print(all_results)
  print("👉 Completed, finished main().")

new_iproyal_proxies = [
  # original proxies (5)
  'socks5://14affc78050af:bd4bb8fa31@185.60.144.62:12324',
  'socks5://14affc78050af:bd4bb8fa31@185.60.144.158:12324',
  'socks5://14affc78050af:bd4bb8fa31@185.60.144.196:12324',
  'socks5://14affc78050af:bd4bb8fa31@185.60.144.98:12324',
  'socks5://14affc78050af:bd4bb8fa31@185.60.145.140:12324',
  
  # new proxies (50)
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
