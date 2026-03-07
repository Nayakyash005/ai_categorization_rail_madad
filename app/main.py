from fastapi import FastAPI,UploadFile,File,Form
from app.classifier import classify_complaint
import json 
app = FastAPI(title="RailMadad ai endpoint")
@app.post("/classify")
async def classify(
        text: str = Form(...),
        image: UploadFile = File(None),
):
    
    image_description = None
    try:

        if image:
            image_description = "dummy"
            # we do something here later 
        response = classify_complaint(complaint_text=text,image_description=image_description)
        print(f" response is {response}")
        return json.loads(response)
    except:
        return {
            "category":"Other",
            "subCategory":"Other",
            "severity":"Low",
            "reason":"Parsing Failed!"
            }

