# app.py
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import torch
from diffusers import StableDiffusionPipeline
import base64
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Serve static files and templates
app.mount("/static", StaticFiles(directory="."), name="static")
templates = Jinja2Templates(directory=".")

# Configuration
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model once at startup
logger.info(f"Loading {MODEL_NAME} on {DEVICE.upper()} with {DTYPE} precision...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=DTYPE)
    pipe = pipe.to(DEVICE)
    logger.info("✅ Model loaded successfully! Ready for generation.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_image(
    prompt: str = Form(...),
    steps: int = Form(30),           # Default: 30 (faster)
    guidance: float = Form(7.5),     # Default guidance
    width: int = Form(512),          # Can be changed via frontend
    height: int = Form(512)
):
    if not prompt.strip():
        return JSONResponse({"error": "Prompt cannot be empty"}, status_code=400)

    # Clamp values to valid ranges
    steps = max(10, min(100, steps))
    guidance = max(1.0, min(20.0, guidance))
    width = 256 if width < 256 else (512 if width <= 512 else 768)
    height = 256 if height < 256 else (512 if height <= 512 else 768)

    logger.info(f"🎨 Generating: '{prompt}' | {width}x{height} | Steps: {steps} | Guidance: {guidance}")

    try:
        with torch.no_grad():  # Save memory
            with torch.cuda.amp.autocast() if DEVICE == "cuda" else nullcontext():
                image = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height
                ).images[0]

        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return {"image": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        logger.error(f"❌ Error generating image: {e}")
        return JSONResponse({"error": "Failed to generate image"}, status_code=500)

# Dummy context for CPU
class nullcontext:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass

# Run server directly (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)