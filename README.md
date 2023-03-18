# 📃 Intuition
I'm building my own multi media GPT; a competitor to [Merlot Reserve](https://rowanzellers.com/merlotreserve/) & [Vid2Seq](https://arxiv.org/abs/2302.14115). It's pre-trained from scratch on youtube data, mostly the YT-1B dataset of 20M curated youtube videos containing significant spoken language (English only).

Project highlights & intuition with photos, check it out: https://twitter.com/KastanDay/status/1595991960380411905

![(No 3D) VPT Architecture Diagram](https://user-images.githubusercontent.com/13607221/212575351-17d9b963-1f33-47cc-a298-db36e9814411.png)

My design follows the "Embedding + Trunk + Head" pattern I first noticed succeeding in DETER and Alphafold2. Now in early 2023, it's successful in [PALM-E](https://arxiv.org/abs/2303.03378) and [Vid2Seq](https://arxiv.org/abs/2302.14115) from Google, and [Prismer](https://arxiv.org/abs/2303.02506) from Nvidia, and many more listed on my Twitter announcement.

<img src="https://user-images.githubusercontent.com/13607221/226117873-3a824b72-9407-4fe1-9cd7-5b969e8a91a2.png" width="800">

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

Install is complete!

## Progress 
1. (Oct 2022) Start of project.
2. (Dec 2022) MVP completed, but messed up the evaluation.
3. (Dec 2022) Migrated all data to Deeplake database library, overall much cleaner & more reliable for distributed database updates.
4. (Jan 2023) Migrated all training logic to Composer, by MosaicML. Super cool library for efficient LLM training, even of huggingface models.
5. (Jan 2023) Finished scaling up distributed pre-processing (i.e. inference w/ Whisper, FlanT5, OpenS and Clip). Rock solid Deeplake distributed `dataset.append()` operations on any size SLURM cluster.
6. (Feb 2023) Tested different backbones: T5 vs T5 v1.1 vs Flan-TS. Somehow, v1.1 was terrible and Flan-T5 was by far the best. As suggested by [another finetuning study](https://twitter.com/ShayneRedford/status/1620805305801261058). The author confirmed this in [my follow-up question](https://twitter.com/KastanDay/status/1620934244372738048).
7. (Mar 2023) WIP: TVQA evaluation. Need to fit more video frames into our 1024 context window, probably by using fewer final hidden states from CLIP.

Up next:

- [ ] Find better scene-graph implementation: conly 55 classes from COCO is not enough for YouTube data. Ours relies on Detectron2 as a base, which is great for in-domain objects but not general. I think the best we can do is to use the 1k classes from imagenet.
- [ ] Totally reimplement sound/audio model to move away from Whisper -- I think [Google's AudioSet](https://research.google.com/audioset/ontology/index.html) with 600+ classes based on YouTube data, will enable the best models. [Here's my favorite](https://github.com/qiuqiangkong/audioset_tagging_cnn#audio-tagging-using-pretrained-models) from that competition.
