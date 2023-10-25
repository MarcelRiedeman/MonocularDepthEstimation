import cv2
import torch
from torchvision.transforms import ToTensor
from zoedepth.utils.misc import colorize
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint
import time

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

print("*" * 20 + " Initializing zoedepth " + "*" * 20)
conf = get_config("zoedepth_nk", "infer")
model = build_model(conf).to(DEVICE)
model.eval()

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    orig_size = frame.shape[:2][::-1]
    X = ToTensor()(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model.infer(X).cpu()

    pred = colorize(out[0])

    # Resize prediction to match the input frame size
    pred = cv2.resize(pred, orig_size)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time

    # Display FPS in the frame
    cv2.putText(pred, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output frame
    cv2.imshow('Depth Prediction', pred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()