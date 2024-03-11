from PIL import Image
import requests
from io import BytesIO
from diffusers import DiffusionPipeline

# Assuming your pipeline setup and image generation as per your code
pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

#prompt = "a photo of an astronaut riding a horse on mars"
#prompt = "a photo of an astronaut carrying tennis ball"
prompt = "a photo of an astronaut carrying a coach bag"
result = pipe(prompt)

# Convert the generated image to a PIL Image object
image = result.images[0]

# Save the image to a file
#image.save("astronaut_riding_horse_on_mars.jpeg", "JPEG")
#image.save("astronaut_tennis_ball.jpeg", "JPEG")
image.save("astronaut_coach_bag.jpeg", "JPEG")

# If you want to save it in another format, just change the file extension and format accordingly. For example, for PNG:
#image.save("astronaut_riding_horse_on_mars.PNG", "PNG")
#image.save("astronaut_tennis_ball.png", "PNG")
image.save("astronaut_coach_bag.png", "PNG")
