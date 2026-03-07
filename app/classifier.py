import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

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

def classify_complaint(complaint_text:str,image_description:str = None) -> dict:

    complaint = complaint_text
    if image_description:
        complaint += f"\n ImageDescription {image_description}"
    prompt = f"""
        you are currently AI supporting Railways related queries ONLY
        You Must have to choose One of the category and subCategory from the following I have Given to You
        Category_Json = {json.dumps(categories_json,indent=2)}
        Description: {complaint_text}
        only answer from the categories that has been given to you do not create your oen category and if description doesnt match with anyone then just return Invalid description
        but do not guess 
        and return output only in this given JSON Format 
        output format:
        {{
            "category": "<exact key from JSON>",
            "subCategory": "<exact value from that category list>",
            "severity": <any of the following enum["high","low","medium"]>,
            "reason": "short explanation"
        }}
    """

    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    model="openai/gpt-oss-120b",
    )
    return chat_completion.choices[0].message.content


print(classify_complaint("TT talking rudely"))