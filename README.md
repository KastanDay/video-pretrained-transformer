# 📃 Intuition
I'm building my own multi media GPT; a competitor to [Merlot Reserve](https://rowanzellers.com/merlotreserve/) & [X-CLIP](https://huggingface.co/docs/transformers/model_doc/xclip). It's pre-trained from scratch on youtube data, mostly the YT-1B dataset of 20M curated youtube videos containing significant spoken language (english only).

![(No 3D) VPT Architecture Diagram](https://user-images.githubusercontent.com/13607221/212575351-17d9b963-1f33-47cc-a298-db36e9814411.png)

I posted a thread on Twitter describing the intuition (with lots of photos), check it out here: https://twitter.com/KastanDay/status/1595991960380411905


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
# An update command might be necessary to get all of ffmpeg's codec-specifc extensions, which we need. 
# solves error in parallel_whisper.py: ❌❌Error during whisper: Expecting value: line 1 column 1 (char 0)
conda update ffmpeg
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

## Progress
1. (Oct 2022) Start of project.
2. (Dec 2022) MVP completed, but messed up the evaluation. 
3. (Dec 2022) Migrated all data to Deeplake database library, overall much cleaner & more reliable than working with raw numpy arrays.
4. (Jan 2023) Migrated all training logic to Composer, by MosaicML. Super cool library for efficient LLM training, even of huggingface models.
5. (Jan 2023) WIP: Finish scaling up distributed pre-processing (i.e. inference w/ Whisper, FlanT5, OpenPSG and Clip)

Todo:

- [ ] Fix evaluation on VQA benchmark; up next: TVQA.
- [ ] Find better scene-graph implementation: just 55ish COCO classes is not enough, best we can do is 1k imagenet classes I think. 
- [ ] Totally reimplement sound/audio model.
