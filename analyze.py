import torch

from deeponet.model import DeepONet, VAONet, TFONet, KANON
from deeponet.data import val_dataset
from deeponet.utils import VAEPredictor, Predictor

import numpy as np

import os
import survey
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Choose checkpoint
checkpoints = os.listdir("checkpoints")
chosen = survey.routines.select("Select checkpoint", options=checkpoints)
checkpoint = f"checkpoints/{checkpoints[chosen]}"

# Load the hyperparameters
hparams = json.load(open(f"{checkpoint}/hparams.json", "r"))

# Load the model
# Check if checkpoint contains 'tf' or 'vae' or 'kan'
if "tf" in checkpoint:
    model = TFONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
elif "vae" in checkpoint:
    model = VAONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = VAEPredictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
elif "kan" in checkpoint:
    model = KANON(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )
else:
    model = DeepONet(hparams)
    model.load_state_dict(torch.load(
        f"{checkpoint}/model.pth", map_location=device))
    predictor = Predictor(
        model,
        device=device,
        study_name="DeepONet-Integral",
        run_name=checkpoints[chosen]
    )

# ==============================================================================
# Validation dataset
# ==============================================================================
# Load the data
ds_val = val_dataset("normal")

for i in range(5):
    u, y, Guy = ds_val[i]
    x = np.linspace(0, 1, len(u))

    # Plot the input function
    predictor.input_plot(x, u, name=f"{i:03}_u_val")

    # Plot the prediction
    predictor.predict_plot(u, y, Guy, name=f"{i:03}_Guy_val")

# ==============================================================================
# Custom test
# ==============================================================================
