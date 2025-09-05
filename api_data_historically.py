from config import API_KEY
import requests

response = requests.get("https://api.nasa.gov/planetary/apod?api_key=" + API_KEY + "&start_date=2025-09-01&end_date=2025-09-05")