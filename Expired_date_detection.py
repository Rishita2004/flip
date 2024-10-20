import pytesseract
import cv2
import re
from pytesseract import Output
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create a Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Prompt the user to upload an image file
file_path = filedialog.askopenfilename(title="Select an image file", 
                                        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

if not file_path:
    print("No file selected.")
else:
    # Load the image from the selected file
    image = cv2.imread(file_path)

    # Preprocessing the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    dilated = cv2.dilate(thresh, None, iterations=1)

    # Apply OCR
    custom_config = r'--oem 3 --psm 6'
    details = pytesseract.image_to_data(dilated, output_type=Output.DICT, config=custom_config, lang='eng')

    # Extracted text
    extracted_text = " ".join(details['text'])
    print("Extracted text:", extracted_text)  # Debugging output

    # Date patterns to match different formats
    date_patterns = [
        r'(\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}/\d{1,2}/\d{1,2}\b)',  # dd/mm/yyyy or yyyy/mm/dd
        r'(\b\w+\s\d{4}\b)',  # e.g., "June 2024"
        r'(\b\d{1,2}\s\w+\s\d{4}\b)',  # e.g., "10 June 2024"
    ]

    dates = []
    # Extract all dates found in the text
    for pattern in date_patterns:
        found_dates = re.findall(pattern, extracted_text)
        dates.extend(found_dates)

    # Convert extracted date strings to datetime objects
    date_objects = []
    for date_str in dates:
        for fmt in ['%d/%m/%Y', '%Y/%m/%d', '%B %Y', '%d %B %Y']:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                date_objects.append(date_obj)
                break  # Exit the loop if a date was successfully parsed
            except ValueError:
                continue

    # Determine MFG and EXP dates
    if date_objects:
        mfg_date = min(date_objects)
        exp_date = max(date_objects)
    else:
        mfg_date = None
        exp_date = None

    # Logic to detect only decimal numbers as MRP values
    mrp_details = []
    # Updated pattern to match decimal numbers only
    decimal_pattern = r'(?<!\d)(\d+\.\d{1,2})(?!\d)'

    # Find all matches for decimal amounts
    matches = re.findall(decimal_pattern, extracted_text)

    # Collect the amounts, removing any potential surrounding whitespace
    for amount in matches:
        mrp_details.append(amount.strip())

    # Remove duplicates
    mrp_details = list(set(mrp_details))  # Remove duplicates

    # Print results
    print("Detected MFG date:", mfg_date.strftime('%d/%m/%Y') if mfg_date else 'Not found')
    print("Detected EXP date:", exp_date.strftime('%d/%m/%Y') if exp_date else 'Not found')
    print("Detected MFG date (formatted):", mfg_date.strftime('%B %Y') if mfg_date else 'Not found')
    print("Detected EXP date (formatted):", exp_date.strftime('%B %Y') if exp_date else 'Not found')
    print("Detected MRP details:", mrp_details if mrp_details else 'Not found')
