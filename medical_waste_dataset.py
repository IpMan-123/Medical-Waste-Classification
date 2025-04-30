import pandas as pd

# Create a more comprehensive dataset
data = {
    'Description': [
        # Sharps
        "needle", "syringe", "scalpel", "lancet", "blade",
        "broken glass", "sharp instrument", "surgical blade", "iv needle", "suture needle",
        "blood collection needle", "dermatome", "biopsy needle", "disposable scalpel", "endoscopic blade",
        "insulin needle", "dental needle", "laparoscopic blade", "surgical chisel", "surgical probe",
        "injection needle", "vascular blade", "surgical scissors", "surgical hook", "surgical staple",
        "disposable lancet", "sterile blade", "surgical pin", "neurosurgical blade", "microtome blade",
        "microsurgical blade", "suture removal scissors", "surgical wire", "electrosurgical blade", "vaccination needle",
        "catheter needle", "surgical drill", "amputation knife", "surgical trocar", "surgical elevator",
        "surgical rasp", "surgical needle", "suture needle holder", "ophthalmic blade", "surgical curette",
        "surgical clamp", "blood lancet", "suture cutter", "safety needle", "surgical saw",

        # Chemical
        "chemical waste", "acid", "toxic chemical", "laboratory chemical", "chemical container",
        "hazardous chemical", "chemical solution", "chemical reagent", "chemical residue", "chemical filter",
        "chemical centrifuge", "chemical purifier", "chemical powder", "chemical liquid", "chemical dispenser",
        "chemical apron", "chemical cooler", "disinfectant", "cleaning agent", "chemical spill",
        "chemical thermometer", "chemical absorbent", "chemical extractor", "chemical beaker", "corrosive chemical",
        "chemical label", "solvent", "chemical vapor", "chemical test kit", "chemical pipette",
        "chemical hazard sign", "chemical heater", "oxidizing agent", "chemical sensor", "chemical flask",
        "chemical titrant", "flammable chemical", "chemical mixer", "chemical indicator", "chemical scrubber",
        "chemical reagent bottle", "chemical balance", "chemical fume hood", "chemical splash goggles", "chemical storage cabinet",
        "chemical safety data sheet", "chemical glove", "chemical neutralizer", "chemical mask", "radioactive material",

        # Pharmaceutical
        "medicine bottle", "pill container", "pharmaceutical", "medication", "expired drug",
        "unused medicine", "drug vial", "medicine packaging", "vaccine", "injectable drug",
        "pharmacy waste", "pharmacy vial", "pharmacy leaflet", "pharmacy inventory", "pharmacy label",
        "pharmacy refrigerator", "brand-name drug", "tablet", "antipyretic", "pharmacy blister pack",
        "over-the-counter drug", "pharmacy bottle", "antibiotic", "syrup", "antiseptic",
        "pharmacy prescription", "analgesic", "ointment", "inhaler", "prescription drug",
        "controlled substance", "pharmacy box", "generic drug", "pharmacy container", "capsule",
        "eye drops", "nasal spray", "lozenge", "suppository", "cream",
        "pharmacy return", "pharmacy packaging", "pharmacy counter", "pharmacy shelf", "pharmacy management",
        "pharmacy assistant", "pharmacy technician", "pharmacy research", "pharmacy education", "pharmacy system",

        # General
        "rubber glove", "rubber gloves", "latex glove", "latex gloves", "medical glove",
        "medical gloves", "surgical glove", "surgical gloves", "protective glove", "protective gloves",
        "examination glove", "examination gloves", "nitrile glove", "nitrile gloves", "disposable glove",
        "disposable gloves", "bandage", "gauze", "cotton swab", "paper towel",
        "packaging", "plastic container", "disposable item", "dressing", "medical packaging",
        "alcohol swab", "face mask", "surgical mask", "face shield", "N95 mask",
        "sterile wrap", "shoe cover", "head cover", "disposable apron", "sterile towel",
        "medical tape", "adhesive bandage", "hot pack", "cold pack", "ice pack",
        "disposable sterile drape", "disposable sterile towel", "disposable shoe cover", "disposable hair cover", "hair cover",
        "protective gown", "medical forceps", "medical scissors", "stethoscope", "hand sanitizer"


    ],
    'Category': [
        # Sharps (50)
        *['Sharps'] * 50,
        
        # Chemical (50)
        *['Chemical'] * 50,
        
        # Pharmaceutical (50)
        *['Pharmaceutical'] * 50,

        # General (50)
        *['General'] * 50
    ]
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('medical_waste_dataset.csv', index=False)