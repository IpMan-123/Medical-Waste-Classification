import pandas as pd

# Create a more comprehensive dataset
data = {
    'Description': [
        # Sharps
        "needle", "syringe", "scalpel", "lancet", "blade",
        "broken glass", "sharp instrument", "surgical blade", "iv needle", "suture needle",
        
        # Chemical
        "chemical waste", "acid", "toxic chemical", "laboratory chemical",
        "chemical container", "hazardous chemical", "chemical solution", "chemical reagent",
        
        # Pharmaceutical
        "medicine bottle", "pill container", "pharmaceutical", "medication",
        "expired drug", "unused medicine", "drug vial", "medicine packaging",
        
        # General
        "rubber glove", "rubber gloves", "latex glove", "latex gloves",
        "rubber glove", "rubber gloves", "latex glove", "latex gloves",
        "medical glove", "medical gloves", "surgical glove", "surgical gloves",
        "protective glove", "protective gloves", "examination glove", "examination gloves",
        "nitrile glove", "nitrile gloves", "disposable glove", "disposable gloves",
        "bandage", "gauze", "cotton swab", "paper towel", "packaging",
        "plastic container", "disposable item", "dressing", "medical packaging"

    ],
    'Category': [
        # Sharps (8)
        *['Sharps'] * 10,
        
        # Chemical (8)
        *['Chemical'] * 8,
        
        # Pharmaceutical (8)
        *['Pharmaceutical'] * 8,

        # General (29)
        *['General'] * 29
    ]
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('medical_waste_dataset.csv', index=False)