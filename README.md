# Network Bending of Diffusion Models for Audio-Visual Generation
### Luke Dzwonczyk, Carmine Emanuele Cella, and David Ban
### DAFx 2024

In this paper we present the first steps towards the creation of a tool which enables artists to create music visualizations using pre-trained, generative, machine learning models. First, we investigate the application of network bending, the process of applying transforms within the layers of a generative network, to image generation diffusion models by utilizing a range of point-wise, tensor-wise, and morphological operators. We identify a number of visual effects that result from various operators, including some that are not easily recreated with standard image editing tools. We find that this process allows for a continuous, fine-grain control of image generation which can be helpful for creative applications. Next, we generate music-reactive videos using Stable Diffusion by passing audio features as parameters to network bending operators. Finally, we comment on certain transforms which radically shift the image and the possibilities of learning more about the latent space of Stable Diffusion based on these transforms.

Read the paper here: https://www.dafx.de/paper-archive/2024/papers/DAFx24_paper_24.pdf

## Installation instructions:

### Dependencies:
- ffmpeg

### Instructions for M1 Mac

The following steps are taken from https://replicate.com/blog/run-stable-diffusion-on-m1-mac

1. Install Python 3.10 or above and [FFmpeg](https://ffmpeg.org/)
2. Clone this repository (https://github.com/dzluke/Sound-Diffusion)
3. Setup and activate a virtual environment
```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```
4. Install dependencies: `pip install -r requirements.txt`

If you're seeing errors like `Failed building wheel for onnx` you might need to install these packages: 
`brew install Cmake protobuf rust`

5. Download pre-trained model at https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
   - Download `sd-v1-4.ckpt` (~4 GB) on that page. Create a new folder `models/ldm/stable-diffusion-v1` and save the
   model you downloaded as `models/ldm/stable-diffusion-v1/model.ckpt`

### For Windows users

1. Follow instructions for mac until part 3.
2. Set up virtual environment:
```
pip install virtualenv
python -m venv venv
.\venv\Scripts\activate
```

4. Install dependencies: `pip install -r requirementswindows.txt`
  Note that assumes the user has CUDA. If your computer does not have a GPU, you will have to reinstall pytorch without it.

5. Continue following instructions for mac starting at 5.

## Running the code:

`imgen.py`: for generating images with different prompts and network bending functions

`vidgen.py`: for generating videos using txt2img

`audio2video.py`: for generating videos using img2img

`util.py`: contains helper functions and network bending functions

