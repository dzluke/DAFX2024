from pathlib import Path
import librosa
import generate
from subprocess import run
import time
import util
from omegaconf import OmegaConf


# set constants
SAMPLING_RATE = 44100
IMAGE_STORAGE_PATH = Path("./image_outputs")
OUTPUT_VIDEO_PATH = Path("./video_outputs")
util.set_sampling_rate(SAMPLING_RATE)

# initialize paths
OUTPUT_VIDEO_PATH.mkdir(exist_ok=True)
IMAGE_STORAGE_PATH.mkdir(exist_ok=True)
util.clear_dir(IMAGE_STORAGE_PATH)

# helper functions

tic = time.time()

# take input from command line args
args = util.run_argparse()

audio_path = Path(args.audio)

# load input audio
audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE)

# load model
config = OmegaConf.load(f"{args.config}")
model = generate.load_model_from_config(config, f"{args.ckpt}")

# calculate number of frames needed
frame_length = 1. / args.fps  # length of each frame in seconds
num_frames = int(audio.size // (frame_length * SAMPLING_RATE)) + 1

print(f">>> Generating {num_frames} frames")

# the user inputs a visual aesthetic in the form of image(s), video(s), or text

# if the user input is text
input = "a floating orb"
layer = 1
args.seed = 46
encoding = model.get_learned_conditioning(input)
# generate the first frame
generate.text2img(model, input, IMAGE_STORAGE_PATH, args)


# generate frames
for i in range(1, num_frames):
    slice_start = int(i * frame_length * SAMPLING_RATE)
    slice_end = int((i + 1) * frame_length * SAMPLING_RATE)
    audio_slice = audio[slice_start:slice_end]

    # bend is a function that defines how to apply network bending given a latent tensor and audio
    bend = util.subtract_full
    bend_function_name = bend.__name__
    audio_feature = util.skewness
    audio_feature_name = audio_feature.__name__
    audio_feature = audio_feature(audio_slice, SAMPLING_RATE) / 10.
    print(">>> Audio Feature Value:", audio_feature)
    bend = bend(audio_feature)

    init_img_path = IMAGE_STORAGE_PATH / f"{(i - 1):05}.png"  # use the previous image

    args.seed += 1

    generate.img2img(model, encoding, init_img_path, IMAGE_STORAGE_PATH, bend, layer, args)

    # every 10 seconds, create an in progress video
    if i % (10 * args.fps) == 0:
        ffmpeg_command = ["ffmpeg",
                          "-y",  # automatically overwrite if output exists
                          "-framerate", str(args.fps),  # set framerate
                          "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                          "-i", str(audio_path),  # set audio path
                          "-vcodec", "libx264",
                          "-pix_fmt", "yuv420p",
                          "in_progress.mp4"]
        run(ffmpeg_command)


video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{args.fps}fps.mp4"
counter = 1
while video_name.exists():
    video_name = OUTPUT_VIDEO_PATH / f"{audio_path.stem}_{bend_function_name}_{audio_feature_name}_layer{layer}_{args.fps}fps{counter}.mp4"
    counter += 1

# turn images into video
ffmpeg_command = ["ffmpeg",
                  "-y",  # automatically overwrite if output exists
                  "-framerate", str(args.fps),  # set framerate
                  "-i", str(IMAGE_STORAGE_PATH) + "/%05d.png",  # set image source
                  "-i", str(audio_path),  # set audio path
                  "-vcodec", "libx264",
                  "-pix_fmt", "yuv420p",
                  str(video_name)]
run(ffmpeg_command)

print(">>> Generated {} images".format(num_frames))
print(">>> Took", util.time_string(time.time() - tic))
print(">>> Avg time per frame: ", util.time_string((time.time() - tic) / num_frames))
print("Done.")
