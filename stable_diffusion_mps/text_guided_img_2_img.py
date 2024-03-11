from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
import torch
import pickle
import os

nodes_file = "stable-diffusion-2-1-unclip.pickle"
if os.path.exists(nodes_file):
  with open(nodes_file, 'rb') as f:
    pipe = pickle.load(f)
else:
  pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
  )
  pipe = pipe.to("mps")
  with open(nodes_file, 'wb') as f:
    pickle.dump(pipe, f)


url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
init_image = load_image(url)

prompt = "A fantasy landscape, with a coach bag"
images = pipe(init_image, prompt=prompt).images

#images = pipe(init_image).images
for ii in range(len(images)):
    images[ii].save(f"variation_image{ii}.png")
