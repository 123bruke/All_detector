import os

SECRET_KEY = "private-secret-key"

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

SQLALCHEMY_DATABASE_URI = "sqlite:///database/app.db"
SQLALCHEMY_TRACK_MODIFICATIONS = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)
