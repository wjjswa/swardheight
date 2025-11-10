import os
import uuid
from flask import Flask, request, jsonify
import torch
import utils
# import from your existing file
from darts_test import load_single_file, prediction, FixedCNN

import logging, sys, traceback
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)


# ------------ Config (env overridable) ------------
GENOTYPE_PATH = os.getenv("GENOTYPE_PATH", "best_sward_pt.json")
WEIGHTS_PATH  = os.getenv("WEIGHTS_PATH",  "best_sward_model.pth")
IN_CHANNELS   = int(os.getenv("IN_CHANNELS", 1))   # 1 for DSM, 3 for RGB
FC_OUTPUT     = int(os.getenv("FC_OUTPUT",   1))   # regression = 1
N_LAYERS      = int(os.getenv("N_LAYERS",    3))
N_NODES       = int(os.getenv("N_NODES",     2))
C_WIDTH       = int(os.getenv("C_WIDTH",    16))
MIN_VAL       = float(os.getenv("MIN_VAL", "13.444444444444445"))
MAX_VAL       = float(os.getenv("MAX_VAL", "30.88888888888889"))

# ------------ App / Model bootstrap ------------
app = Flask(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Build architecture once
genotype = utils.load_genotype(GENOTYPE_PATH)
model = FixedCNN(genotype, C_in=IN_CHANNELS, C=C_WIDTH,
                 n_classes=FC_OUTPUT, n_layers=N_LAYERS, n_nodes=N_NODES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.to(device)
model.eval()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sward Height Prediction API is running."})

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Accept a file field named 'image'
    if "image" not in request.files:
        return jsonify({"error": "No image file provided (field name should be 'image')."}), 400

    file = request.files["image"]

    # Save to /tmp (writable in Cloud Run) because load_single_file expects a path
    tmp_path = f"/tmp/{uuid.uuid4().hex}.img"
    file.save(tmp_path)

    try:
        img_tensor = load_single_file(tmp_path, MIN_VAL, MAX_VAL, device=device)
        pred_tensor = prediction(img_tensor, model, MIN_VAL, MAX_VAL, device)
        pred_value = float(pred_tensor.squeeze().item())
        return jsonify({"predicted_height": round(pred_value, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    # Cloud Run listens on 8080 by default
    app.run(host="0.0.0.0", port=8080)
