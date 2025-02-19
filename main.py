from fastapi import FastAPI, UploadFile, File
from typing import Optional
import base64 #base64: Converts image binary data into a Base64 string
import requests #requests: Used to fetch the image from a URL
from openai import OpenAI #OpenAI API client, which allows us to interact with GPT-4
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
client = OpenAI()
app = FastAPI()

@app.post("/process-image/")
async def process_image(file: Optional[UploadFile] = File(None)):
    try:
        # if file.content_type not in ["image/png", "image/jpeg"]:
        #     return JSONResponse(content={"error": "Invalid file type. Only PNG and JPG are allowed."}, status_code=400)
        base64_image = encode_image(file.file)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content

     
    except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to process uploaded file: {str(e)}"})

def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')


# huging face 
@app.post("/process-image-huggingface/")
async def process_image_huggingface(file: UploadFile = File(...)):
    # Load the processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Read image file
    image_bytes = await file.read()
    raw_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)

    # Return response as JSON
    return JSONResponse({
        "conditional_caption": conditional_caption,
        "unconditional_caption": unconditional_caption
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8100, reload=True)
