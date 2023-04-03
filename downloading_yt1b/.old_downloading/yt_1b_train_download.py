import ray 
import subprocess
import shlex

def prep_bash_commands():
  golden_command = """yt-dlp -f 'bv*[height<=360]+ba/b[height<=480]' \
  -P /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_{process_num} \
  --batch-file /home/kastan/thesis/video-pretrained-transformer/downloading_formalized/train_id_collection/yt_1b_train_id_list_{process_num}.txt \
  --download-archive /mnt/storage_hdd/thesis/yt_1b_dataset/yt_1b_train/parallel_{process_num}/yt_1b_train_download_record_{process_num}.txt \
  -P 'temp:/tmp' \
  -N 5 \
  -o '%(id)s_%(channel)s_%(view_count)s_%(title)s.%(ext)s' \
  --min-views 500 \
  --proxy  \
  --write-subs"""
  final_subprocess_commands = []
  for i in range(10):
    # set process number, split into cmd list for subprocess
    final_subprocess_commands.append(shlex.split(golden_command.format(process_num=i)))
  print("Verify (first) command is correct:")
  print(shlex.join(final_subprocess_commands[0]))
  print("Num parallel downloads: ", len(final_subprocess_commands))
  return final_subprocess_commands

def main():
  """ MAIN """
  ray.shutdown()
  ray.init()

  @ray.remote
  def persistent_download(subprocess_command):
    result = subprocess.run(subprocess_command)
    return result
  
  # Launch four parallel square tasks.
  final_subprocess_commands = prep_bash_commands()
  futures = [persistent_download.remote(cmd) for cmd in final_subprocess_commands]

  # Retrieve results.
  all_results = ray.get(futures)

  print(len(all_results))
  print(all_results)
  print("👉 Completed, finished main().")

if __name__ == '__main__':
  main()
