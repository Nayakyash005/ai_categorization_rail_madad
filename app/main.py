from fastapi import FastAPI, UploadFile, File, Form
from app.classifier import classify_complaint
from app.embed_classifier import get_query_embedding, find_best_subcategory_match, load_category_embeddings
from app.estimate_severity import find_severity
import json
import tempfile
import os
import requests
import io
from PIL import Image
app = FastAPI(title="RailMadad ai endpoint")

# Load category embeddings once at startup
category_embeddings = load_category_embeddings()

@app.post("/classify_with_llm")
async def classify_with_llm(
    text: str = Form(...),
    image_url: str = Form(None),
):
    image_description = None  # For now, do text-only inference
    try:
        # Note: image_url is accepted but not used for LLM inference yet
        response = classify_complaint(complaint_text=text, image_description=image_description)
        print(f" response is {response}")
        parsed_response = json.loads(response)
        # Ensure the response has the expected keys
        if not all(key in parsed_response for key in ["category", "subCategory", "severity", "reason"]):
            raise ValueError("Invalid response format from LLM")
        return parsed_response
    except:
        return {
            "category": "Other",
            "subCategory": "Other",
            "severity": "low",
            "reason": "Parsing Failed!"
        }


@app.post("/classify")
async def classify_with_embeddings(
    text: str = Form(...),
    image_url: str = Form(None),
):
    try:
        image_path = None
        if image_url:
            # Fetch image from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()  # Raise error for bad status
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image.save(temp_file, format="JPEG")
                image_path = temp_file.name

        # Get query embedding (fuses if image provided)
        query_emb = get_query_embedding(text=text, image_path=image_path)

        # Find best match
        best_match = find_best_subcategory_match(query_emb, category_embeddings)
       
        # Clean up temp file if created
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        # Add severity and reason
        # severity = "medium"
        # if best_match["score"] > 0.9:
        #     severity = "high"
        # elif best_match["score"] < 0.7:
        #     severity = "low"
        severity = find_severity(best_match["subCategory"],best_match["category"])
        reason = f"Matched via embedding similarity ({best_match['score']:.2f})"

        return {
            "category": best_match["category"],
            "subCategory": best_match["subCategory"],
            "severity": severity,
            "reason": reason
        }
    except Exception as e:
        print(f"Error in classify_with_embeddings: {e}")
        return {
            "category": "Other",
            "subCategory": "Other",
            "severity": "low",
            "reason": "Embedding classification failed!"
        }

