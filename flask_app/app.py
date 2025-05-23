from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import random

from src.modeling.modeling import model_QB
from src.preds_infr.preds import predict_by_customer_id

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"xlsx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global variables to store model and data
latest_model = None
latest_journey_df = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/model", methods=["POST"])
def upload():
    global latest_model, latest_journey_df
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        model, report, importance_table_html, journey_df = model_QB(filepath)

        latest_model = model
        latest_journey_df = journey_df
        customer_ids = latest_journey_df['customer_id'].drop_duplicates().sample(n=5, random_state=42).tolist() \
            if len(latest_journey_df) >= 5 else latest_journey_df['customer_id'].drop_duplicates().tolist()
        
        return jsonify({
            "report": report,
            "importance_table_html": importance_table_html,
            "sample_customer_ids": customer_ids
        })
    else:
        return jsonify({"error": "Invalid file type. Please upload an XLSX file."})

@app.route("/predict_by_customer", methods=["GET"])
def predict_by_customer():
    global latest_model, latest_journey_df
    customer_id = request.args.get("customer_id")
    json_values = predict_by_customer_id(latest_model, latest_journey_df, customer_id)
    return json_values

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)