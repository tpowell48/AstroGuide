from config import API_KEY
import importlib
import requests
import json
from datetime import datetime, timedelta
import os
import sys

START_DATE = datetime(2022, 12, 15)
END_DATE = datetime(2025, 9, 10)
API_URL = 'http://127.0.0.1:8000/v1/apod/'
OUTPUT_FILENAME = 'apod_data.json'

# Add the current directory to Python's path to ensure it finds 'apod-api'
sys.path.insert(0, os.getcwd())

try:
    # Import the apod_object_parser.py file directly
    apod_module = importlib.import_module("apod-api.apod_parser.apod_object_parser")
    
    print("Successfully imported the 'apod_object_parser' module!")

except ImportError as e:
    print(f"ImportError: Could not import the module. {e}")
    print("Ensure 'apod-api' and 'apod_parser' directories have __init__.py files.")
except AttributeError as e:
    print(f"AttributeError: The module was found, but an object was not inside it: {e}")


def fetch_in_chunks(start_date, end_date, api_key):
    """
    Fetches APOD data in monthly chunks to avoid timeouts.
    """
    all_data = []
    current_start = start_date
    
    while current_start <= end_date:
        # Set the end of the chunk to one month later, or the overall end date
        current_end = current_start + timedelta(days=30)
        if current_end > end_date:
            current_end = end_date
            
        print(f"Fetching data from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}...")
        
        params = {
            # 'api_key': api_key,
            'start_date': current_start.strftime('%Y-%m-%d'),
            'end_date': current_end.strftime('%Y-%m-%d'),
        }
        
        try:
            response = requests.get(API_URL, params=params, timeout=60) # Add a timeout
            response.raise_for_status()
            all_data.extend(response.json())
            
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for chunk starting {current_start.strftime('%Y-%m-%d')}. Skipping.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            
        # Move to the next chunk
        current_start = current_end + timedelta(days=1)
        
    return all_data

def save_image(all_data):
    """
    Downloads and saves an image from a URL.
    """
    for img in all_data:
        try:
            apod_module.download_image(img['url'], img['date'], directory="APOD_DATA/IMAGES")
        except KeyError:
            print(f"No image 'url' found for date {img.get('date', 'unknown')}. Skipping image download.")



# --- Run the script ---
if __name__ == "__main__":
    final_data = fetch_in_chunks(START_DATE, END_DATE, API_KEY)
    
    if final_data:
        with open("APOD_DATA/" + OUTPUT_FILENAME, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"\nSuccessfully downloaded and saved all data to {OUTPUT_FILENAME}")
    else:
        print("\nNo data was downloaded.")
    
    save_image(final_data)
    


