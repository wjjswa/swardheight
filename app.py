from flask import Flask, request, jsonify
import torch
from darts_test import load_single_file, test_model_train_iterate

# --------------------------------------------------------
# Flask app initialization
# --------------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------------
# Model and configuration setup (runs once at container start)
# --------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# paths (relative to container)
genotype = "data/best_sward_pt.json"
model_path = "data/best_sward_model.pth"

# dataset normalization parameters (adjust if known)
min_val = 13.444444444444445
max_val = 30.88888888888889
in_channels = 1
fc_output = 1

print("✅ Model configuration loaded successfully.")

# --------------------------------------------------------
# Routes
# --------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Sward Height Estimation API is running.",
        "usage": "POST an image to /predict"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded. Use 'image' field."}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        # Convert file bytes to temporary tensor
        with open("/tmp/uploaded_image.jpg", "wb") as f:
            f.write(image_bytes)

        # preprocess image
        img_tensor = load_single_file("/tmp/uploaded_image.jpg", min_val, max_val, device=device)

        # run model prediction
        test_preds = test_model_train_iterate(
            genotype,
            img_tensor,
            min_val,
            max_val,
            in_channels,
            fc_output,
            device,
        )

        predicted_value = float(test_preds.cpu().numpy().flatten()[0])

        return jsonify({
            "predicted_height": round(predicted_value, 4),
            "unit": "cm (example)",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
