# multi-media-GPT
Building my own version of OPT-175B + CLIP + others. Pre-trained from scratch on youtube data.

# 🚀 Quickstart

1. Install Git LFS
```bash
# Install `git-lfs` (via apt or brew)
brew install git-lfs
-OR-
conda install -c conda-forge -y git-lfs
```
Then start GitLFS
```bash
git-lfs install
```

2. Install ffmpeg

A simple install should work fine, despite how convoluted the library tends to be.
```bash
# preffered
sudo apt update && sudo apt install ffmpeg
-OR-
# conda method is not as well tested for this project
conda install -c conda-forge -y ffmpeg
```

3. Clone the repo with our custom submodules
```bash
git clone --recurse-submodules git@github.com:KastanDay/video-pretrained-transformer.git
```

4. Install pip requirements 
```bash
pip install -r ./requirements.txt
```

Later, if updates are made to submodules, you can pull new changes using:
```bash
git submodule update --remote
```

We have submodules in the first place because we needed to modify the internal logic of three libraries used in preprocessing: Lhotse (to be faster), OpenPSG, and Transformers to modify the T5 implementation to suport modality encodings.

Done!
