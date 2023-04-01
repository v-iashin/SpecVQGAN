# Taming Visually Guided Sound Generation

BMVC 2021 – Oral Presentation

• [[Project Page](https://v-iashin.github.io/SpecVQGAN)]
• [[ArXiv](http://arxiv.org/abs/2110.08791)]
• [[BMVC Proceedings](https://www.bmvc2021-virtualconference.com/conference/papers/paper_1213.html)]
• [[Poster (for PAISS)](https://v-iashin.github.io/images/specvqgan/poster.pdf)]
• [[Presentation on YouTube](https://www.youtube.com/watch?v=Bucb3nAa398)] ([Can't watch YouTube?](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/SpecVQGAN%20YouTube.mp4))
•

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pxTIMweAKApJZ3ZFqyBee3HtMqFpnwQ0?usp=sharing)

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/specvqgan/specvqgan_vggsound_samples.jpg" alt="Generated Samples Using our Model" width="900">

Listen for the samples on our [project page](https://v-iashin.github.io/SpecVQGAN).

# Overview
We propose to tame the visually guided sound generation by shrinking a training dataset to a set of representative vectors aka. a codebook.
These codebook vectors can, then, be controllably sampled to form a novel sound given a set of visual cues as a prime.

The codebook is trained on spectrograms similarly to [VQGAN](https://arxiv.org/abs/2012.09841) (an upgraded [VQVAE](https://arxiv.org/abs/1711.00937)).
We refer to it as **Spectrogram VQGAN**

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/specvqgan/codebook.svg" alt="Spectrogram VQGAN" width="900">

Once the spectrogram codebook is trained, we can train a **transformer** (a variant of [GPT-2](https://openai.com/blog/better-language-models/)) to autoregressively sample the codebook entries as tokens conditioned on a set of visual features

<img src="https://github.com/v-iashin/v-iashin.github.io/raw/master/images/specvqgan/transformer.svg" alt="Vision-based Conditional Cross-modal Autoregressive Sampler" width="900">

This approach allows training a spectrogram generation model which produces long, relevant, and high-fidelity sounds while supporting tens of data classes.

- [Taming Visually Guided Sound Generation](#taming-visually-guided-sound-generation)
- [Overview](#overview)
- [Environment Preparation](#environment-preparation)
  - [Conda](#conda)
  - [Docker](#docker)
- [Data](#data)
  - [Download](#download)
  - [Extract Features Manually](#extract-features-manually)
- [Pretrained Models](#pretrained-models)
  - [Codebooks](#codebooks)
  - [Transformers](#transformers)
  - [VGGish-ish, Melception, and MelGAN](#vggish-ish-melception-and-melgan)
- [Training](#training)
  - [Training a Spectrogram Codebook](#training-a-spectrogram-codebook)
  - [Training a Transformer](#training-a-transformer)
    - [VAS Transformer](#vas-transformer)
    - [VGGSound Transformer](#vggsound-transformer)
    - [Controlling the Condition Size](#controlling-the-condition-size)
  - [Training VGGish-ish and Melception](#training-vggish-ish-and-melception)
  - [Training MelGAN](#training-melgan)
- [Evaluation](#evaluation)
- [Sampling Tool](#sampling-tool)
- [The Neural Audio Codec Demo](#the-neural-audio-codec-demo)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

<!-- The link to this section is used in demo.ipynb -->
# Environment Preparation

During experimentation, we used Linux machines with `conda` virtual environments, PyTorch 1.8 and CUDA 11.

Start by cloning this repo
```bash
git clone https://github.com/v-iashin/SpecVQGAN.git
```

Next, install the environment.
For your convenience, we provide both `conda` and `docker` environments.

## Conda
```bash
conda env create -f conda_env.yml
```
Test your environment
```bash
conda activate specvqgan
python -c "import torch; print(torch.cuda.is_available())"
# True
```

## Docker
Download the image from Docker Hub and test if CUDA is available:
```bash
docker run \
    --mount type=bind,source=/absolute/path/to/SpecVQGAN/,destination=/home/ubuntu/SpecVQGAN/ \
    --mount type=bind,source=/absolute/path/to/logs/,destination=/home/ubuntu/SpecVQGAN/logs/ \
    --mount type=bind,source=/absolute/path/to/vggsound/features/,destination=/home/ubuntu/SpecVQGAN/data/vggsound/ \
    --shm-size 8G \
    -it --gpus '"device=0"' \
    iashin/specvqgan:latest \
    python
>>> import torch; print(torch.cuda.is_available())
# True
```
or build it yourself
```bash
docker build - < Dockerfile --tag specvqgan
```

# Data
In this project, we used [VAS](https://github.com/PeihaoChen/regnet#download-datasets) and [VGGSound](www.robots.ox.ac.uk/~vgg/data/vggsound/) datasets.
VAS can be downloaded directly using the link provided in the [RegNet](https://github.com/PeihaoChen/regnet#download-datasets) repository.
For VGGSound, however, one might need to retrieve videos directly from YouTube.

## Download
The scripts will download features, check the `md5` sum, unpack, and do a clean-up for each part of the dataset:
```bash
cd ./data
# 24GB
bash ./download_vas_features.sh
# 420GB (+ 420GB if you also need ResNet50 Features)
bash ./download_vggsound_features.sh
```
The unpacked features are going to be saved in `./data/downloaded_features/*`.
Move them to `./data/vas` and `./data/vggsound` such that the folder structure would match the structure of the demo files.
By default, it will download `BN Inception` features, to download `ResNet50` features uncomment the lines in scripts `./download_*_features.sh`

If you wish to download the parts manually, use the following URL templates:

- `https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vas/*.tar`
- `https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggsound/*.tar`

Also, make sure to check the `md5` sums provided in [`./data/md5sum_vas.md5`](./data/md5sum_vas.md5) and [`./data/md5sum_vggsound.md5`](./data/md5sum_vggsound.md5) along with file names.

Note, we distribute features for the VGGSound dataset in 64 parts.
Each part holds ~3k clips and can be used independently as a subset of the whole dataset (the parts are not class-stratified though).

## Extract Features Manually

For `BN Inception` features, we employ the same procedure as [RegNet](https://github.com/PeihaoChen/regnet#data-preprocessing).

For `ResNet50` features, we rely on [video_features (branch `specvqgan`)](https://github.com/v-iashin/video_features/tree/specvqgan)
repository and used these commands:
```bash
# VAS (few hours on three 2080Ti)
strings=("dog" "fireworks" "drum" "baby" "gun" "sneeze" "cough" "hammer")
for class in "${strings[@]}"; do
    python main.py \
        --feature_type resnet50 \
        --device_ids 0 1 2 \
        --batch_size 86 \
        --extraction_fps 21.5 \
        --file_with_video_paths ./paths_to_mp4_${class}.txt \
        --output_path ./data/vas/features/${class}/feature_resnet50_dim2048_21.5fps \
        --on_extraction save_pickle
done

# VGGSound (6 days on three 2080Ti)
python main.py \
    --feature_type resnet50 \
    --device_ids 0 1 2 \
    --batch_size 86 \
    --extraction_fps 21.5 \
    --file_with_video_paths ./paths_to_mp4s.txt \
    --output_path ./data/vggsound/feature_resnet50_dim2048_21.5fps \
    --on_extraction save_pickle
```
Similar to `BN Inception`, we need to "tile" (cycle) a video if it is shorter than 10s. For
`ResNet50` we achieve this by tiling the resulting frame-level features up to 215 on temporal dimension, e.g. as follows:
```python
feats = pickle.load(open(path, 'rb')).astype(np.float32)
reps = 1 + (215 // feats.shape[0])
feats = np.tile(feats, (reps, 1))[:215, :]
with open(new_path, 'wb') as file:
    pickle.dump(feats, file)
```

<!-- <details>
<summary>Downloading VGGSound from Scratch</summary>

1. We will rely on the AudioSet download script. To adapt it, we refactor `vggsound.csv` using the following script such that can be used in a AudioSet downloader:

```python
import pandas as pd

VGGSOUND_PATH = './data/vggsound.csv'
VGGSOUND_REF_PATH = './data/vggsound_ref.csv'
vggsound_meta = pd.read_csv(VGGSOUND_PATH, names=['YTID', 'start_seconds', 'positive_labels', 'split'])
vggsound_meta['end_seconds'] = vggsound_meta['start_seconds'] + 10
vggsound_meta = vggsound_meta.drop(['split'], axis=1)
vggsound_meta = vggsound_meta[['YTID', 'start_seconds', 'end_seconds', 'positive_labels']]
print(list(vggsound_meta.columns))
print(vggsound_meta.head())
vggsound_meta.to_csv(VGGSOUND_REF_PATH, sep=',', index=None, header=None)
```

1. We also add 3 lines with `# placeholder` on top of the `vggsound_ref.csv` to match the style as AudioSet has
some statistics there.
1. Rent an instance (GoogleCloud/AWS/Pouta), allocate an IP. Disk 800GB: 300 GB for video and 90 for audio + zipping + OS
1. `git clone https://github.com/marl/audiosetdl` and check out to `ebd89c5` commit.  This code provides a script to download AudioSet in parallel on several CPUs.
1. Create a file with conda environment in `down_audioset.yaml` with content as follows:

```yaml
name: down_audioset
channels:
  - conda-forge
  - yaafe
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - bzip2=1.0.8=h7b6447c_0
  - ca-certificates=2020.6.24=0
  - certifi=2020.6.20=py38_0
  - ffmpeg=4.2.2=h20bf706_0
  - freetype=2.10.2=h5ab3b9f_0
  - gmp=6.1.2=h6c8ec71_1
  - gnutls=3.6.5=h71b1129_1002
  - lame=3.100=h7b6447c_0
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libedit=3.1.20191231=h7b6447c_0
  - libffi=3.3=he6710b0_2
  - libflac=1.3.1=0
  - libgcc-ng=9.1.0=hdf63c60_0
  - libogg=1.3.2=0
  - libopus=1.3.1=h7b6447c_0
  - libpng=1.6.37=hbc83047_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libvpx=1.7.0=h439df22_0
  - mad=0.15.1b=he1b5a44_0
  - ncurses=6.2=he6710b0_1
  - nettle=3.4.1=hbb512f6_0
  - openh264=2.1.0=hd408876_0
  - openssl=1.1.1g=h516909a_0
  - pip=20.1.1=py38_1
  - python=3.8.3=hcff3b4d_2
  - python_abi=3.8=1_cp38
  - readline=8.0=h7b6447c_0
  - setuptools=47.3.1=py38_0
  - sqlite=3.32.3=h62c20be_0
  - tk=8.6.10=hbc83047_0
  - wheel=0.34.2=py38_0
  - x264=1!157.20191217=h7b6447c_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - pip:
    - cffi==1.14.0
    - multiprocessing-logging==0.2.4
    - numpy==1.19.0
    - pafy==0.5.3.1
    - pycparser==2.20
    - pysoundfile==0.9.0.post1
    - scipy==1.5.1
    - sk-video==1.1.8
    - sox==1.3.3
    - youtube-dl==2020.6.16.1
```

1. Install the environment: `conda env create -f down_audioset.yml`
1. Open `download_audioset.py` in the cloned repo (`audiosetdl`) and comment the following lines as AudioSet has 3 parts:

```python
    # download_subset(balanced_train_segments_path, data_dir, ffmpeg_path, ffprobe_path,
    #                 num_workers, **ffmpeg_cfg)
    # download_subset(unbalanced_train_segments_path, data_dir, ffmpeg_path, ffprobe_path,
    #                 num_workers, **ffmpeg_cfg))
```

6. `(down_audioset) $ python download_audioset.py ./vggsound -e ./vggsound_ref.csv -n 8 -nr 2 -f $(which ffmpeg) -fp $(which ffprobe) -lp ./log.txt --verbose`
- The script will download the dataset into `./vggsound` folder (mp4 videos and flac audios).
- It will use 8 cores in parallel. Also checkout other arguments
- I observed a trade-off between number of CPUs and how quickly the IP will be banned by YouTube
- during the first pass I changed 3 IPs. Might be because I was using `-nr 10` and too many CPUs (8) or
    because I started downloading vids from the same csv -> same videos were checked again -> too many requests
- I found it useful to shuffle the csv data
`cp vggsound_ref.csv vggsound_noshuf.csv && shuf -o vggsound_ref.csv vggsound_ref_notshuf.csv`
Remember you have 3 lines on top of `vggsound_ref.csv` which are now suffled? Find them and put on top.
- Don't run it from your local machine!
- The authors of VGGsound used .wav audios but the script downloads .flac. You may want to adjust the script
    such that it will do it automatically. You may also somehow skip audios and just extracted them later from
    .mp4.
- It will take some time. In my case it was one week, including reruns on shuffled data.

</details> -->

# Pretrained Models
Unpack the pre-trained models to `./logs/` directory.

## Codebooks
| Trained on | Evaluated on | FID ↓ | Avg. MKL ↓ |                                                                                                                                                          Link / MD5SUM |
| ---------: | -----------: | ----: | ---------: | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   VGGSound |     VGGSound |   1.0 |        0.8 | [7ea229427297b5d220fb1c80db32dbc5](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-05-19T22-16-54_vggsound_codebook.tar.gz) |
|        VAS |          VAS |   6.0 |        1.0 |      [0024ad3705c5e58a11779d3d9e97cc8a](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-06T19-42-53_vas_codebook.tar.gz) |

Run [Sampling Tool](#sampling-tool) to see the reconstruction results for available data.

## Transformers

The setting **(a)**: the transformer is trained on *VGGSound* to sample from the *VGGSound* codebook:
<!-- NOTE: Mind the streamlit visualization as well if more models will be released -->
| Condition |     Features | FID ↓ | Avg. MKL ↓ | Sample Time️ ↓ |                                                                                                                                                             Link / MD5SUM |
| --------: | -----------: | ----: | ---------: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  No Feats |              |  13.5 |        9.7 |           7.7 | [b1f9bb63d831611479249031a1203371](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-20T16-35-20_vggsound_transformer.tar.gz) |
|    1 Feat | BN Inception |   8.6 |        7.7 |           7.7 | [f2fe41dab17e232bd94c6d119a807fee](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T11-18-51_vggsound_transformer.tar.gz) |
|    1 Feat |     ResNet50 | 11.5* |       7.3* |           7.7 | [27a61d4b74a72578d13579333ed056f6](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-30T21-03-22_vggsound_transformer.tar.gz) |
|   5 Feats | BN Inception |   9.4 |        7.0 |           7.9 | [b082d894b741f0d7a1af9c2732bad70f](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T09-34-10_vggsound_transformer.tar.gz) |
|   5 Feats |     ResNet50 | 11.3* |       7.0* |           7.9 | [f4d7105811589d441b69f00d7d0b8dc8](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-30T21-34-25_vggsound_transformer.tar.gz) |
| 212 Feats | BN Inception |   9.6 |        6.8 |          11.8 | [79895ac08303b1536809cad1ec9a7502](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T07-27-58_vggsound_transformer.tar.gz) |
| 212 Feats |     ResNet50 | 10.5* |       6.9* |          11.8 | [b222cc0e7aeb419f533d5806a08669fe](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-30T21-34-41_vggsound_transformer.tar.gz) |

\* – calculated on 1 sample per video the test set instead of 10 samples per video that is used for the rest.
Evaluating a model on a larger number of samples per video is an expensive procedure.
When evaluated on 10 samples per video, one might expect that the values might improve a bit (~+0.1).

The setting **(b)**: the transformer is trained on *VAS* to sample from the *VGGSound* codebook
| Condition |     Features | FID ↓ | Avg. MKL ↓ | Sample Time️ ↓ |                                                                                                                                                        Link / MD5SUM |
| --------: | -----------: | ----: | ---------: | ------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  No Feats |              |  33.7 |        9.6 |           7.7 | [e6b0b5be1f8ac551700f49d29cda50d7](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-20T16-34-36_vas_transformer.tar.gz) |
|    1 Feat | BN Inception |  38.6 |        7.3 |           7.7 | [a98a124d6b3613923f28adfacba3890c](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T06-32-51_vas_transformer.tar.gz) |
|    1 Feat |     ResNet50 | 26.5* |       6.7* |           7.7 | [37cd48f06d74176fa8d0f27303841d94](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T11-47-40_vas_transformer.tar.gz) |
|   5 Feats | BN Inception |  29.1 |        6.9 |           7.9 | [38da002f900fb81275b73e158e919e16](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T05-51-34_vas_transformer.tar.gz) |
|   5 Feats |     ResNet50 | 22.3* |       6.5* |           7.9 | [7b6951a33771ef527f1c1b1f99b7595e](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T11-36-00_vas_transformer.tar.gz) |
| 212 Feats | BN Inception |  20.5 |        6.0 |          11.8 | [1c4e56077d737677eac524383e6d98d3](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T05-38-40_vas_transformer.tar.gz) |
| 212 Feats |     ResNet50 | 20.8* |       6.2* |          11.8 | [6e553ea44c8bc7a3310961f74e7974ea](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T11-52-28_vas_transformer.tar.gz) |

\* – calculated on 10 samples per video the test set instead of 100 samples per video that is used for the rest.
Evaluating a model on a larger number of samples per video is an expensive procedure.
When evaluated on 10 samples per video, one might expect that the values might improve a bit (~+0.1).

The setting **(c)**: the transformer is trained on *VAS* to sample from the *VAS* codebook
| Condition |     Features | FID ↓ | Avg. MKL ↓ | Sample Time ↓ |                                                                                                                                                        Link / MD5SUM |
| --------: | -----------: | ----: | ---------: | ------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|  No Feats |              |  28.7 |        9.2 |           7.6 | [ea4945802094f826061483e7b9892839](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-20T16-24-38_vas_transformer.tar.gz) |
|    1 Feat | BN Inception |  25.1 |        6.6 |           7.6 | [8a3adf60baa049a79ae62e2e95014ff7](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-09T13-31-37_vas_transformer.tar.gz) |
|    1 Feat |     ResNet50 | 25.1* |       6.3* |           7.6 | [a7a1342030653945e97f68a8112ed54a](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T14-59-49_vas_transformer.tar.gz) |
|   5 Feats | BN Inception |  24.8 |        6.2 |           7.8 | [4e1b24207780eff26a387dd9317d054d](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-09T14-14-24_vas_transformer.tar.gz) |
|   5 Feats |     ResNet50 | 20.9* |       6.1* |           7.8 | [78b8d42be19dd1b0a346b1f512967302](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T14-51-25_vas_transformer.tar.gz) |
| 212 Feats | BN Inception |  25.4 |        5.9 |          11.6 | [4542632b3c5bfbf827ea7868cedd4634](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-09T15-17-18_vas_transformer.tar.gz) |
| 212 Feats |     ResNet50 | 22.6* |       5.8* |          11.6 | [dc2b5cbd28ad98d2f9ca4329e8aa0f64](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-07-29T13-34-39_vas_transformer.tar.gz) |

\* – calculated on 10 samples per video the test set instead of 100 samples per video that is used for the rest.
Evaluating a model on a larger number of samples per video is an expensive procedure.
When evaluated on 10 samples per video, one might expect that the values might improve a bit (~+0.1).

A transformer can also be trained to generate a spectrogram given a specific **class**.
We also provide pre-trained models for all three settings:
The setting **(c)**: the transformer is trained on *VAS* to sample from the *VAS* codebook
| Setting | Codebook | Sampling for | FID ↓ | Avg. MKL ↓ | Sample Time ↓ |                                                                                                                                                             Link / MD5SUM |
| ------: | -------: | -----------: | ----: | ---------: | ------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|     (a) | VGGSound |     VGGSound |   7.8 |        5.0 |           7.7 | [98a3788ab973f1c3cc02e2e41ad253bc](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-03T00-43-28_vggsound_transformer.tar.gz) |
|     (b) | VGGSound |          VAS |  39.6 |        6.7 |           7.7 |      [16a816a270f09a76bfd97fe0006c704b](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-08T14-41-19_vas_transformer.tar.gz) |
|     (c) |      VAS |          VAS |  23.9 |        5.5 |           7.6 |      [412b01be179c2b8b02dfa0c0b49b9a0f](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/models/2021-06-09T09-42-07_vas_transformer.tar.gz) |

## VGGish-ish, Melception, and MelGAN

These will be downloaded automatically during the first run.
However, if you need them separately, here are the checkpoints
- [VGGish-ish](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggishish16.pt) (1.54GB, `197040c524a07ccacf7715d7080a80bd`) + Normalization Parameters (in `/specvqgan/modules/losses/vggishish/data/`)
- [Melception](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melception-21-05-10T09-28-40.pt) (0.27GB, `a71a41041e945b457c7d3d814bbcf72d`) + Normalization Parameters (in `/specvqgan/modules/losses/vggishish/data/`)
- [MelGAN](./vocoder/logs/vggsound). If you wish to continue training it here are checkpoints
[netD.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melgan_ckpt/netD.pt),
[netG.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melgan_ckpt/netG.pt),
[optD.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melgan_ckpt/optD.pt),
[optG.pt](https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melgan_ckpt/optG.pt).


The reference performance of VGGish-ish and Melception:
| Model      | Top-1 Acc | Top-5 Acc | mAP   | mAUC  |
| ---------- | --------- | --------- | ----- | ----- |
| VGGish-ish | 34.70     | 63.71     | 36.63 | 95.70 |
| Melception | 44.49     | 73.79     | 47.58 | 96.66 |

Run [Sampling Tool](#sampling-tool) to see Melception and MelGAN in action.

# Training
The training is done in **two** stages.
First, a **spectrogram codebook** should be trained.
Second, a **transformer** is trained to sample from the codebook
The first and the second stages can be trained on the same or separate datasets as long as the process of spectrogram extraction is the same.

## Training a Spectrogram Codebook

> **Erratum**: during training with the default config, the code will silently fail to load the checkpoint of
> the perceptual loss. This leads to the results which are as good as without the perceptual loss.
> For this reason, one may try turning it off completely: `perceptual_weight=0.0` and benefit from faster
> iterations. For details please refer to [Issue#13](https://github.com/v-iashin/SpecVQGAN/issues/13)

To train a spectrogram codebook, we tried two datasets: VAS and VGGSound.
We run our experiments on a relatively expensive hardware setup with four _40GB NVidia A100_ but the models
can also be trained on one _12GB NVidia 2080Ti_ with smaller batch size.
When training on four _40GB NVidia A100_, change arguments to `--gpus 0,1,2,3` and
`data.params.batch_size=8` for the codebook and `=16` for the transformer.
The training will hang a bit at `0, 2, 4, 8, ...` steps because of the logging.
If folders with features and spectrograms are located elsewhere, the paths can be specified in
`data.params.spec_dir_path`, `data.params.rgb_feats_dir_path`, and `data.params.flow_feats_dir_path`
arguments but use the same format as in the config file e.g. notice the `*`
in the path which globs class folders.

```bash
# VAS Codebook
# mind the comma after `0,`
python train.py --base configs/vas_codebook.yaml -t True --gpus 0,
# or
# VGGSound codebook
python train.py --base configs/vggsound_codebook.yaml -t True --gpus 0,
```

## Training a Transformer
A transformer (GPT-2) is trained to sample from the spectrogram codebook given a set of frame-level visual features.

### VAS Transformer

```bash
# with the VAS codebook
python train.py --base configs/vas_transformer.yaml -t True --gpus 0, \
    model.params.first_stage_config.params.ckpt_path=./logs/2021-06-06T19-42-53_vas_codebook/checkpoints/epoch_259.ckpt
# or with the VGGSound codebook which has 1024 codes
python train.py --base configs/vas_transformer.yaml -t True --gpus 0, \
    model.params.transformer_config.params.GPT_config.vocab_size=1024 \
    model.params.first_stage_config.params.n_embed=1024 \
    model.params.first_stage_config.params.ckpt_path=./logs/2021-05-19T22-16-54_vggsound_codebook/checkpoints/epoch_39.ckpt
```

### VGGSound Transformer

```bash
python train.py --base configs/vggsound_transformer.yaml -t True --gpus 0, \
    model.params.first_stage_config.params.ckpt_path=./logs/2021-05-19T22-16-54_vggsound_codebook/checkpoints/epoch_39.ckpt
```

### Controlling the Condition Size
The size of the visual condition is controlled by two arguments in the config file.
The `feat_sample_size` is the size of the visual features resampled equidistantly from all available features (212) and `block_size` is the attention span.
Make sure to use `block_size = 53 * 5 + feat_sample_size`.
For instance, for `feat_sample_size=212` the `block_size=477`.
However, the longer the condition, the more memory and more timely the sampling.
By default, the configs are using `feat_sample_size=212` for VAS and `5` for VGGSound.
Feel free to tweak it to your liking/application for example:
```bash
python train.py --base configs/vas_transformer.yaml -t True --gpus 0, \
    model.params.transformer_config.params.GPT_config.block_size=318 \
    data.params.feat_sampler_cfg.params.feat_sample_size=53 \
    model.params.first_stage_config.params.ckpt_path=./logs/2021-06-06T19-42-53_vas_codebook/checkpoints/epoch_259.ckpt
```
The `No Feats` settings (without visual condition) are trained similarly to the settings with visual conditioning where the condition is replaced with random vectors.
The optimal approach here is to use `replace_feats_with_random=true` along with `feat_sample_size=1` for example (VAS):
```bash
python train.py --base configs/vas_transformer.yaml -t True --gpus 0, \
    data.params.replace_feats_with_random=true \
    model.params.transformer_config.params.GPT_config.block_size=266 \
    data.params.feat_sampler_cfg.params.feat_sample_size=1 \
    model.params.first_stage_config.params.ckpt_path=./logs/2021-06-06T19-42-53_vas_codebook/checkpoints/epoch_259.ckpt
```

## Training VGGish-ish and Melception
We include all necessary files for training both `vggishish` and `melception` in `./specvqgan/modules/losses/vggishish`.
Run it on a 12GB GPU as
```bash
cd ./specvqgan/modules/losses/vggishish
# vggish-ish
python train_vggishish.py config=./configs/vggish.yaml device='cuda:0'
# melception
python train_melception.py config=./configs/melception.yaml device='cuda:0'
```

## Training MelGAN
To train the vocoder, use this command:
```bash
cd ./vocoder
python scripts/train.py \
    --save_path ./logs/`date +"%Y-%m-%dT%H-%M-%S"` \
    --data_path /path/to/melspec_10s_22050hz \
    --batch_size 64
```

# Evaluation
The evaluation is done in two steps.
First, the samples are generated for each video. Second, evaluation script is run.
The sampling procedure supports multi-gpu multi-node parallization.
We provide a multi-gpu command which can easily be applied on a multi-node setup by replacing `--master_addr` to your main machine and `--node_rank` for every worker's id (also see an `sbatch` script in `./evaluation/sbatch_sample.sh` if you have a SLURM cluster at your disposal):
```bash
# Sample
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
        evaluation/generate_samples.py \
        sampler.config_sampler=evaluation/configs/sampler.yaml \
        sampler.model_logdir=$EXPERIMENT_PATH \
        sampler.splits=$SPLITS \
        sampler.samples_per_video=$SAMPLES_PER_VIDEO \
        sampler.batch_size=$SAMPLER_BATCHSIZE \
        sampler.top_k=$TOP_K \
        data.params.spec_dir_path=$SPEC_DIR_PATH \
        data.params.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
        data.params.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
        sampler.now=$NOW
# Evaluate
python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=62374 \
    --use_env \
    evaluate.py \
        config=./evaluation/configs/eval_melception_${DATASET,,}.yaml \
        input2.path_to_exp=$EXPERIMENT_PATH \
        patch.specs_dir=$SPEC_DIR_PATH \
        patch.spec_dir_path=$SPEC_DIR_PATH \
        patch.rgb_feats_dir_path=$RGB_FEATS_DIR_PATH \
        patch.flow_feats_dir_path=$FLOW_FEATS_DIR_PATH \
        input1.params.root=$EXPERIMENT_PATH/samples_$NOW/$SAMPLES_FOLDER
```
The variables for the **VAS** dataset:
```bash
EXPERIMENT_PATH="./logs/<folder-name-of-vas-transformer-or-codebook>"
SPEC_DIR_PATH="./data/vas/features/*/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="./data/vas/features/*/feature_rgb_bninception_dim1024_21.5fps/"
FLOW_FEATS_DIR_PATH="./data/vas/features/*/feature_flow_bninception_dim1024_21.5fps/"
SAMPLES_FOLDER="VAS_validation"
SPLITS="\"[validation, ]\""
SAMPLER_BATCHSIZE=4
SAMPLES_PER_VIDEO=10
TOP_K=64 # use TOP_K=512 when evaluating a VAS transformer trained with a VGGSound codebook
NOW=`date +"%Y-%m-%dT%H-%M-%S"`
```
The variables for the **VGGSound** dataset:
```bash
EXPERIMENT_PATH="./logs/<folder-name-of-vggsound-transformer-or-codebook>"
SPEC_DIR_PATH="./data/vggsound/melspec_10s_22050hz/"
RGB_FEATS_DIR_PATH="./data/vggsound/feature_rgb_bninception_dim1024_21.5fps/"
FLOW_FEATS_DIR_PATH="./data/vggsound/feature_flow_bninception_dim1024_21.5fps/"
SAMPLES_FOLDER="VGGSound_test"
SPLITS="\"[test, ]\""
SAMPLER_BATCHSIZE=32
SAMPLES_PER_VIDEO=1
TOP_K=512
NOW=`date +"%Y-%m-%dT%H-%M-%S" the`
```

# Sampling Tool
For interactive sampling, we rely on the [Streamlit](https://streamlit.io/) library.
To start the streamlit server locally, run
```bash
# mind the trailing `--`
streamlit run --server.port 5555 ./sample_visualization.py --
# go to `localhost:5555` in your browser
```
or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pxTIMweAKApJZ3ZFqyBee3HtMqFpnwQ0?usp=sharing).

We also alternatively provide a similar notebook in `./generation_demo.ipynb` to play with the demo on
a local machine.

# The Neural Audio Codec Demo
While the Spectrogram VQGAN was never designed to be a neural audio codec but
it happened to be highly effective for this task.
We can employ our Spectrogram VQGAN pre-trained on an open-domain dataset as a
neural audio codec without a change

If you wish to apply the SpecVQGAN for audio compression for arbitrary audio,
please see our Google Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K_-e6CRQFLk9Uq6O46FOsAYt63TeEdXf?usp=sharing).

Integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). See demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/SpecVQGAN_Neural_Audio_Codec)

We also alternatively provide a similar notebook in `./neural_audio_codec_demo.ipynb` to play with the demo on
a local machine.

# Citation
Our paper was accepted as an oral presentation for the BMVC 2021.
Please, use this bibtex if you would like to cite our work
```
@InProceedings{SpecVQGAN_Iashin_2021,
  title={Taming Visually Guided Sound Generation},
  author={Iashin, Vladimir and Rahtu, Esa},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2021}
}
```

# Acknowledgments
Funding for this research was provided by the Academy of Finland projects 327910 & 324346. The authors acknowledge CSC — IT Center for Science, Finland, for computational resources for our experimentation.

We also acknowledge the following work:
- The code base is built upon an amazing [taming-transformers](https://github.com/CompVis/taming-transformers) repo.
Check it out if you are into high-res image generation.
- The implementation of some evaluation metrics is partially borrowed and adapted from [torch-fidelity](https://github.com/toshas/torch-fidelity).
- The feature extraction pipeline for BN-Inception relies on the baseline implementation [RegNet](https://github.com/PeihaoChen/regnet).
- MelGAN training scripts are built upon the [official implementation for text-to-speech MelGAN](https://github.com/descriptinc/melgan-neurips).
- Thanks [AK391](https://github.com/AK391) for adapting our neural audio codec demo as a
Gradio app at [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/SpecVQGAN_Neural_Audio_Codec)
