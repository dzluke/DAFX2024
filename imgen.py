import argparse
from pathlib import Path
import librosa
import torch
import math
from generate import text2img, load_model_from_config
import numpy as np
from subprocess import run
import time
import util
from omegaconf import OmegaConf


parser = argparse.ArgumentParser()
parser.add_argument(
    "--skip_grid",
    action='store_true',
    default=True,
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action='store_true',
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--plms",
    action='store_true',
    help="use plms sampling",
)
parser.add_argument(
    "--laion400m",
    action='store_true',
    help="uses the LAION400M model",
)
parser.add_argument(
    "--fixed_code",
    action='store_true',
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--config",
    type=str,
    default="configs/stable-diffusion/v1-inference.yaml",
    help="path to config which constructs model",
)
parser.add_argument(
    "--ckpt",
    type=str,
    default="models/ldm/stable-diffusion-v1/model.ckpt",
    help="path to checkpoint of model",
)
parser.add_argument(
    "--seed",
    type=int,
    default=43,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
args = parser.parse_args()

# set constants
IMAGE_STORAGE_PATH = Path("./image_outputs")

# initialize paths
IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
util.clear_dir(IMAGE_STORAGE_PATH)

# helper functions
tic = time.time()


# load model
config = OmegaConf.load(f"{args.config}")
model = load_model_from_config(config, f"{args.ckpt}")

# the layer to perform network bending at
# layer = 0 means apply before the first layer
# layer = 1 means apply after the first layer
# layer = 0

# scales = [10**x for x in range(-5, 6)]  # for binary threshold
# scales = [10**x for x in range(6)]
# scales = [0] + scales
# scales = [10**x for x in range(-1, 3)]
# e = [-1, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
# scales = [10**x for x in e]
# scales = [10**x for x in range(4)]
# scales = [x * math.pi for x in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]]
# scales = [10, 30, 50, 70, 80, 90, 100, 1000, 10000]
# scales = [0.8, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.2]
# scales = [0, 0.8, 0.9, 0.95, 0.99, 1, 1.01, 1.05, 1.1, 1.2, 10]

# Generates a list of tuples, where each tuple contains two values. Ex: [(-1, -0.9), (-0.9, -0.8), ...]
increment = 0.5
start_value = -1
end_value = 1
scales = [(i, i + increment) for i in [round(x, 1) for x in  [start_value + j*increment for j in range(int((end_value - start_value) / increment) + 1)]]]


num_imgs = len(scales)

prompt = "a floating orb"

# dims = [0, 1, 2]

layers = [0, 10, 20, 30, 40, 49]
# layers = [0, 1, 2, 3, 4, 5]

print(f">>> Generating {num_imgs} images")

# generate frames
# for d in dims:
for l in layers:
    folder = IMAGE_STORAGE_PATH / f'layer{l}'
    folder.mkdir(parents=True, exist_ok=True)
    for s in scales:
        args.seed = 46  # use this seed for cool orb pic
        # args.seed += 1
        # bend is a function that defines how to apply network bending given a latent tensor
        # bend = util.apply_to_dim(util.threshold, s, d, 32)

        # bend = util.apply_to_dim(util.add_full, s, d, 32)
        # bend = util.normalize2(lambda x: x)
        bend = util.clamp(s)

        text2img(model, prompt, folder, args, bend, l)


print(">>> Generated {} images".format(num_imgs))
print(">>> Took", util.time_string(time.time() - tic))
print(">>> Avg time per img: ", util.time_string((time.time() - tic) / num_imgs))
print("Done.")
