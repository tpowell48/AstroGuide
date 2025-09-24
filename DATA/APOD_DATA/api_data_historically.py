import sys
import os
import importlib
import requests
import json
from datetime import datetime, timedelta
from PIL import Image, UnidentifiedImageError

START_DATE = datetime(2022, 12, 20)
END_DATE = datetime(2025, 9, 15)
API_URL = 'http://127.0.0.1:8000/v1/apod/'
OUTPUT_FILENAME = 'DATA/APOD_DATA/apod_data.json'

# Add the current directory to Python's path to ensure it finds 'apod-api'
sys.path.insert(0, os.getcwd())

try:
    apod_module = importlib.import_module("apod-api.apod_parser.apod_object_parser")
    print("Successfully imported custom download module.")
except (ImportError, AttributeError) as e:
    print(f"Could not import custom download module: {e}.")
    apod_module = None

def fetch_in_chunks(start_date, end_date):
    """Fetches APOD data in monthly chunks."""
    all_data = []
    current_start = start_date
    while current_start <= end_date:
        current_end = current_start + timedelta(days=30)
        if current_end > end_date:
            current_end = end_date
        
        print(f"Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        params = {
            'start_date': current_start.strftime('%Y-%m-%d'),
            'end_date': current_end.strftime('%Y-%m-%d'),
        }
        try:
            response = requests.get(API_URL, params=params, timeout=60)
            response.raise_for_status()
            all_data.extend(response.json())
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        current_start = current_end + timedelta(days=1)
    return all_data

def process_and_clean_data(all_data, filename, IMAGE_DIR='DATA/APOD_DATA/IMAGES'):
    """
    Downloads, verifies, and cleans the data, saving a final JSON file
    with only valid, non-corrupt image entries.
    """
    valid_entries = []
    
    print("\n--- Starting Download and Verification Process ---")
    for item in all_data:
        date = item.get('date')
        url = item.get('url')
        media_type = item.get('media_type')

        # Skip if not an image entry
        if media_type != 'image':
            print(f"Skipping entry for {date}: Not an image (media_type: {media_type})")
            continue

        if not url or not date:
            continue

        image_path = os.path.join(IMAGE_DIR, f"{date}.jpg")

        # Download the image if it doesn't exist
        if not os.path.exists(image_path):
            apod_module.download_image(url, date, directory=IMAGE_DIR)

        # # Verify the image file exists and is not corrupt
        # if os.path.exists(image_path):
        #     try:
        #         with Image.open(image_path) as img:
        #             img.verify()
        #         # If verification passes, add the entry to our clean list
        #         valid_entries.append(item)
        #     except (UnidentifiedImageError, IOError, SyntaxError):
        #         print(f"Corrupted image detected for {date}. Deleting file and skipping entry.")
        #         os.remove(image_path) # Delete the corrupted file
        # else:
        #     print(f"Missing image file for date {date}. Skipping entry.")
            
    # Save the final, cleaned list to the JSON file
    print(f"\nSaving {len(valid_entries)} verified entries to '{filename}'...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=4)
    print("JSON file has been cleaned and saved.")


# --- Run the script ---
if __name__ == "__main__":
    # Fetch all data from the API
    raw_data = fetch_in_chunks(START_DATE, END_DATE)
    
    if raw_data:
        # Process, download, verify, and save the clean data
        process_and_clean_data(raw_data, OUTPUT_FILENAME)
        print("\nProcess complete.")
    else:
        print("\nNo data fetched from the API. Exiting.")