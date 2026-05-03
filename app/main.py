from fastapi import FastAPI, UploadFile, File, Form
from app.classifier import classify_complaint
from app.embed_classifier import get_query_embedding, find_best_subcategory_match, load_category_embeddings
import json
import tempfile
import os

app = FastAPI(title="RailMadad ai endpoint")

# Load category embeddings once at startup
category_embeddings = load_category_embeddings()

@app.post("/classify_with_llm")
async def classify_with_llm(
    text: str = Form(...),
    image: UploadFile = File(None),
):
    image_description = None
    try:
        if image:
            image_description = "dummy"
            # we do something here later
        response = classify_complaint(complaint_text=text, image_description=image_description)
        print(f" response is {response}")
        return json.loads(response)
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
    image: UploadFile = File(None),
):
    try:
        image_path = None
        if image:
            # Save uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(await image.read())
                image_path = temp_file.name

        # Get query embedding (fuses if image provided)
        query_emb = get_query_embedding(text=text, image_path=image_path)

        # Find best match
        best_match = find_best_subcategory_match(query_emb, category_embeddings)

        # Clean up temp file if created
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)

        # Add severity and reason (simple logic for now)
        severity = "medium"  # Default
        if best_match["score"] > 0.9:
            severity = "high"
        elif best_match["score"] < 0.7:
            severity = "low"

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


