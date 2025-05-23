from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for flashing messages

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")


@app.route("/model", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("home"))
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("home"))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        # Read the XLSX file with pandas
        df = pd.read_excel(filepath)
        # Do your modeling/inference here
        # For demonstration, just show the first 5 rows as HTML
        table_html = df.head().to_html()
        return f"<h2>Preview of Uploaded File</h2>{table_html}<br><a href='{url_for('home')}'>Back</a>"
    else:
        flash("Invalid file type. Please upload an XLSX file.")
        return redirect(url_for("home"))

