from flask import Blueprint, request, jsonify, session
from ml.ai_models import predict_image
import os

image = Blueprint("image", __name__)

@image.route("/upload", methods=["POST"])
def upload():
    if "user" not in session:
        return jsonify(error="Login required")

    file = request.files["file"]
    path = os.path.join("uploads", file.filename)
    file.save(path)

    result = predict_image(path)
    return jsonify(result=result)
