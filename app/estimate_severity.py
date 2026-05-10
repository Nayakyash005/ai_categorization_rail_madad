from fastapi import FastAPI, UploadFile, File, Form
import json
import torch

severity_mapping = {

    "medical Assisance": {
        "medical assistance": "high"
    },

    "Security": {
        "Eve-Teasing/Misbehaviour with lady passengers/Rape": "high",
        "Theft of Passengers Belongings/Snatching": "high",
        "Unauthorized person in Ladies/Disabled Coach/SLR/Reserve Coach": "high",
        "Harrasment/Extortion by security Personal/Railway personnel": "high",
        "Nuisance by Hawkers/Beggar/Eunuch": "medium",
        "Luggage Left Behind/Unclaimed/Suspected Articles": "high",
        "Passenger Missing/Not Responding call": "high",
        "Smoking/Drinking Alcohol/Narcotics": "medium",
        "Dacoity/Robbery/Murder/Riots": "high",
        "Quarrelling/Hooliganism": "medium",
        "Passenger fallen down": "high",
        "Nuisance by Passenger": "medium",
        "Misbehaviour": "medium",
        "Others": "medium"
    },

    "Handicapped Facilities": {
        "Handicapped Coach Facilities": "medium",
        "Handicapped toilet/washbasin": "medium",
        "Braille signage in coach": "low",
        "Others": "medium"
    },

    "Facilities for Women with Special needs": {
        "Baby Food": "high"
    },

    "Electrical Equipment": {
        "Air Conditioner": "medium",
        "Fans": "medium",
        "Lights": "medium",
        "Charging Points": "low",
        "Others": "medium"
    },

    "coach-cleanliness": {
        "Toilet": "medium",
        "Washbasin": "medium",
        "Cockroach/Rodents": "high",
        "Coach Interior": "low",
        "Coach Exterior": "low",
        "Others": "medium"
    },

    "Punctuality": {
        "NTES APP": "low",
        "Late Running": "medium",
        "Others": "medium"
    },

    "Water Availability": {
        "Packaged Drinking Water/Rail Neer": "medium",
        "Toilet": "medium",
        "Washbasin": "medium",
        "Others": "medium"
    },

    "coach-maintenance": {
        "window/seat broken": "medium",
        "window/door locking problem": "high",
        "tap leaking/tap not working problem": "medium",
        "broken/missing toilet fittings": "medium",
        "jerks/abnormal sounds": "high",
        "other": "medium"
    },

    "Catering & Vending Services": {
        "Overcharging": "low",
        "Service Quality & Hygiene": "medium",
        "Food Quality & Quantity": "medium",
        "E-Catering": "low",
        "Food & Water Not Available": "high",
        "Others": "medium"
    },

    "staff behavior": {
        "Staff Behaviour": "medium"
    },

    "Corruption/Bribery": {
        "Corruption/Bribery": "high"
    },

    "Bed Roll": {
        "Dirty/Torn": "low",
        "Overcharging": "low",
        "Non Availability": "medium",
        "Others": "medium"
    }
}


def find_severity(sub_category: str, category: str):
    try:
        severity = severity_mapping.get(category,{}).get(sub_category,"medium")
        if (severity  not in ["high","low","medium"]):
            return "medium"
        print("Severity for this complaints is ",severity)
        return severity
    except:
        return "medium"