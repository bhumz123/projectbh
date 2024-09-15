pip install easyocr
import easyocr
import re
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cv2

patterns = {
    'height': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(cm|cms|centimeter|centimeters|ft|feet|foot|in|"|'
        r'inch|inches|m|meter|meters|mm|millimeter|millimeters|yd|yard|yards)\b',
        re.IGNORECASE
    ),
    'width': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(cm|cms|centimeter|centimeters|ft|feet|foot|in|'
        r'inch|"|inches|m|meter|meters|mm|millimeter|millimeters|yd|yard|yards)\b',
        re.IGNORECASE
    ),
    'depth': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(cm|cms|centimeter|centimeters|ft|feet|foot|in|'
        r'inch|"|inches|m|meter|meters|mm|millimeter|millimeters|yd|yard|yards)\b',
        re.IGNORECASE
    ),
    'maximum_weight_recommendation': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(g|gs|gsm|gram|grams|kg|kgs|kilogram|kilograms|'
        r'mg|mgs|milligram|milligrams|mcg|mcgs|microgram|micrograms|oz|ounce|'
        r'ounces|lb|lbs|pound|pounds|ton|tons)\b',
        re.IGNORECASE
    ),
    'item_weight': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(g|gs|gsm|gram|grams|kg|kgs|kilogram|kilograms|'
        r'mg|mgs|milligram|milligrams|mcg|mcgs|microgram|micrograms|oz|ounce|'
        r'ounces|lb|lbs|pound|pounds|ton|tons)\b',
        re.IGNORECASE
    ),
    'voltage': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(kv|kilovolt|mv|millivolt|v|volt|volts)\b',
        re.IGNORECASE
    ),
    'wattage': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(kw|kilowatt|w|watt|watts)\b',
        re.IGNORECASE
    ),
    'item_volume': re.compile(
        r'\b[0-9O]+([-~][0-9O]+)?(\.[0-9O]+)?\s*(cl|centiliter|cu ft|cubic foot|cu in|cubic inch|'
        r'cup|cups|dl|deciliter|fl oz|fl|oz|ounce|ounces|fluid ounce|gal|gallon|'
        r'imp gal|imperial gallon|l|liter|litre|µl|μl|ul|microliter|milliliter|'
        r'milliliters|ml|pt|pint|qt|quart)\b',
        re.IGNORECASE
    )
}

# Create an EasyOCR reader object
reader = easyocr.Reader(['en'])

def is_valid_url(url):
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.RequestException:
        return False

def fetch_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception:
        return None

def apply_ocr_and_match(image, pattern):
    results = reader.readtext(image)
    matches = []
    for (box, text, _) in results:
        if pattern.search(text):
            matches.append((text, box))
    return matches

def process_group(group, pattern):
    extracted_texts = []
    bounding_boxes = []
    for index, image_link in enumerate(group['image_link']):
        print(f"Processing image {index}: {image_link}")
        if is_valid_url(image_link):
            image = fetch_image_from_url(image_link)
            if image is not None:
                matches = apply_ocr_and_match(image, pattern)
                texts = [match[0] for match in matches]
                boxes = [match[1] for match in matches]
                extracted_texts.append(texts)
                bounding_boxes.append(boxes)
            else:
                extracted_texts.append([])
                bounding_boxes.append([])
        else:
            extracted_texts.append([])
            bounding_boxes.append([])
    group['extracted_text'] = extracted_texts
    group['bounding_box'] = bounding_boxes
    return group

# Read the CSV file
df = pd.read_csv('train.csv')

# Group by entity_name and apply the appropriate function
result_df = pd.DataFrame()
for entity_name, pattern in patterns.items():
    group = df[df['entity_name'] == entity_name]
    if not group.empty:
        processed_group = process_group(group, pattern)
        result_df = pd.concat([result_df, processed_group])

# Save the result to a new CSV file
result_df.to_csv('processed_results_testing_6.csv', index=False)
