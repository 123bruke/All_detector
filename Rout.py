from flask import Blueprint, request, jsonify, session
from database.models import User, db

auth = Blueprint("auth", __name__)

@auth.route("/login", methods=["POST"])
def login():
    email = request.form["email"]
    password = request.form["password"]
    user = User.query.filter_by(email=email, password=password).first()

    if user:
        session["user"] = {"id": user.id, "name": user.name}
        return jsonify(success=True)

    return jsonify(success=False)
