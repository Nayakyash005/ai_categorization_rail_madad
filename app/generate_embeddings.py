import torch
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()  # Set to evaluation mode

# Define the categories JSON (same as in classifier.py)
categories_json = {
    "medical Assisance": ["medical assistance"],
    "Security": ["Eve-Teasing/Misbehaviour with lady passengers/Rape", "Theft of Passengers Belongings/Snatching", "Unauthorized person in Ladies/Disabled Coach/SLR/Reserve Coach",
                 "Harrasment/Extortion by security Personal/Railway personnel", "Nuisance by Hawkers/Beggar/Eunuch", "Luggage Left Behind/Unclaimed/Suspected Articles",
                 "Passenger Missing/Not Responding call", "Smoking/Drinking Alcohol/Narcotics", "Dacoity/Robbery/Murder/Riots", "Quarrelling/Hooliganism", 
                 "Passenger fallen down", "Nuisance by Passenger", "Misbehaviour", "Others"],
    "Handicapped Facilities": ["Handicapped Coach Facilities", "Handicapped toilet/washbasin", "Braille signage in coach", "Others"],
    "Facilities for Women with Special needs": ["Baby Food"],
    "Electrical Equipment": ["Air Conditioner", "Fans", "Lights", "Charging Points", "Others"],
    "coach-cleanliness": ["Toilet", "Washbasin", "Cockroach/Rodents", "Coach Interior", "Coach Exterior", "Others"],
    "Punctuality": ["NTES APP", "Late Running", "Others"],
    "Water Availability": ["Packaged Drinking Water/Rail Neer", "Toilet", "Washbasin", "Others"],
    "coach-maintenance": ["window/seat broken", "window/door locking problem", "tap leaking/tap not working problem", 
                     "broken/missing toilet fittings", "jerks/abnormal sounds", "other"],
    "Catering & Vending Services": ["Overcharging", "Service Quality & Hygiene", "Food Quality & Quantity", "E-Catering", "Food & Water Not Available", "Others"],
    "staff behavior": ["Staff Behaviour"],
    "Corruption/Bribery": ["Corruption/Bribery"],
    "Bed Roll": ["Dirty/Torn", "Overcharging", "Non Availability", "Others"]
}

def generate_category_embeddings():
    """
    Generate CLIP text embeddings for all categories and subcategories.
    Returns a dictionary with category -> subcategory -> embedding.
    """
    category_embeddings = {}

    for category, subcategories in categories_json.items():
        category_embeddings[category] = {}
        for subcategory in subcategories:
            # Prepare text input for CLIP (subcategory as text)
            inputs = processor(text=[subcategory], return_tensors="pt", padding=True)
            
            # Generate text embeddings
            with torch.no_grad():
                outputs = model.get_text_features(**inputs)
                text_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                # Normalize the embeddings
                text_features = text_features / torch.norm(text_features, dim=-1, keepdim=True)
            
            # Store as list for JSON serialization
            category_embeddings[category][subcategory] = text_features.squeeze().tolist()

    return category_embeddings

if __name__ == "__main__":
    # Generate embeddings
    embeddings = generate_category_embeddings()
    
    # Save to JSON file
    with open("category_embeddings.json", "w") as f:
        json.dump(embeddings, f, indent=2)
    
    print("Category embeddings generated and saved to category_embeddings.json")