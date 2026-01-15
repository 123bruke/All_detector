from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

@app.route("/")
def home():
    return render_template("bira.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files["image"]
    path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
    image.save(path)

    return jsonify({
        "message": "🖼️ Image uploaded successfully",
        "image_url": "/" + path
    })

if __name__ == "__main__":
    app.run(debug=True)
