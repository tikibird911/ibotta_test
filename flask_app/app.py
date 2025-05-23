from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
import joblib
import uuid


from src.modeling.modeling import model_QB

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
        # Train and get model, report, and fig
        model, report, fig = model_QB(filepath)
        # Save the trained model for later use
        model_path = os.path.join(app.config["UPLOAD_FOLDER"], "latest_xgb_model.pkl")
        joblib.dump(model, model_path)
        # Save the plot if it exists
        plot_url = None
        if fig:
            plot_filename = f"plot_{uuid.uuid4().hex}.png"
            plot_path = os.path.join(app.config["UPLOAD_FOLDER"], plot_filename)
            fig.savefig(plot_path)
            plot_url = url_for("uploaded_file", filename=plot_filename)
        return render_template(
            "model_result.html",
            report=report,
            plot_url=plot_url
        )
    else:
        flash("Invalid file type. Please upload an XLSX file.")
        return redirect(url_for("home"))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)