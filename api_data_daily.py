from config import API_KEY
from apod_api.apod_parser import apod_object_parser
from datetime import date, timedelta

response = apod_object_parser.get_data(API_KEY)