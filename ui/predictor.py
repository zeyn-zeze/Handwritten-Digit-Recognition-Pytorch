# predictor.py
import os
import torch
from model_digit import MNISTModel

device = torch.device("cpu")

def load_model(weights_file="model_digit2.pth") -> MNISTModel:
    model = MNISTModel().to(device)
    if os.path.exists(weights_file):
        state = torch.load(weights_file, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    model.eval()
    return model

def predict(model: MNISTModel, x: torch.Tensor) -> tuple[int, list[float]]:
    with torch.inference_mode():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred = int(torch.argmax(logits, dim=1).item())
    return pred, probs.tolist()
