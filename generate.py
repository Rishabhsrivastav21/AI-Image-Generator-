from diffusers import StableDiffusionPipeline
import torch

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "a beautiful sunset over the mountains"
print(f"Generating: {prompt}")

image = pipe(prompt).images[0]
image.save("ai_image.png")
print("✅ Image saved as ai_image.png")