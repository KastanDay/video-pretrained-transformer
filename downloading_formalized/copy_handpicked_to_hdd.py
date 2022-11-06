import os
import shutil
channel_list = ['wall_street_journal', 'corridor_crew', 'cinemasins', 'matt_davella', 'berm_peak']
channel_human_list = ['Wall Street Journal', 'Corridor Crew', 'CinemaSins', 'Matt D\'Avella', 'Berm Peak']

# reverse order of channel_list 
channel_list = channel_list[::-1]
channel_human_list = channel_human_list[::-1]

root_dir = '/home/kastan/thesis/video-pretrained-transformer/yt_download'
save_dir = '/mnt/storage_hdd/thesis/handpicked_downloads'

file_list = os.listdir(root_dir)

for channel_dir, channel_human_name in zip(channel_list, channel_human_list):
  if not os.path.exists(os.path.join(save_dir, channel_dir)):
    os.makedirs(os.path.join(save_dir, channel_dir), exist_ok=True)
  print("Starting: ", channel_human_name)
  
  for name in file_list:
    if channel_human_name in name:
      # move file from root_dir to save_dir
      if not os.path.exists(os.path.join(save_dir, channel_dir, name)):
        shutil.copy(os.path.join(root_dir, name), os.path.join(save_dir, channel_dir))
    