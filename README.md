# multi-media-GPT
Building my own version of OPT-175B + CLIP + others. Pre-trained from scratch on youtube data.

# Installing

1. Git LFS
```bash
# Install `git-lfs` (via apt or brew)
brew install git-lfs
-OR-
conda install -c conda-forge -y git-lfs
```
Start GitLFS
```bash
git-lfs install
```

2. Clone the repo with our custom submodules
```bash
git clone --recurse-submodules git@github.com:KastanDay/video-pretrained-transformer.git
```

As updates are made to submodules, you can pull new changes using:
```bash
git submodule update --remote
```
Done!
