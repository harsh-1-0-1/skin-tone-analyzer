from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_skin_tone(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Skin detection mask
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    skin = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite("static/uploads/skin_mask.jpg", skin)

    # Average skin color
    avg_color = cv2.mean(image, mask=mask)[:3]  # BGR
    avg_color_rgb = avg_color[::-1]  # convert to RGB

    r, g, b = avg_color_rgb
    if r > g and r > b:
        tone = "Warm"
        suggestions = ["Olive", "Mustard", "Maroon", "Peach"]
    elif b > r and b > g:
        tone = "Cool"
        suggestions = ["Navy", "Emerald", "Lavender", "Gray"]
    else:
        tone = "Neutral"
        suggestions = ["Beige", "White", "Black", "Teal"]

    return tone, suggestions, avg_color_rgb

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            tone, suggestions, avg_color = get_skin_tone(filepath)

            return render_template(
                "result.html",
                image_name=filename,
                tone=tone,
                avg_color=avg_color,
                suggestions=suggestions
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
