import cv2
import torch
from zoedepth.utils.config import get_config
from zoedepth.models.builder import build_model

# ZoeD_NK model initialization
conf = get_config("zoedepth_nk", "infer")
model_zoe_nk = build_model(conf)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_nk.to(DEVICE)
zoe.eval()  # Set the model to evaluation mode

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (512, 384))

    # Convert NumPy array to PyTorch tensor and reshape
    img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(DEVICE)

    # Get the dimensions of the input image
    input_height, input_width = img.shape[:2]

    with torch.no_grad():
        prediction = zoe(img_tensor)

        # Assuming 'metric_depth' key contains the depth tensor
        depth_prediction = prediction['metric_depth']

        # Specify the output size as a tuple (depth_height, depth_width)
        output_size = (depth_prediction.shape[2], depth_prediction.shape[3])

        # Interpolate the depth tensor to match the input image dimensions
        interpolated_depth = torch.nn.functional.interpolate(
            depth_prediction,
            size=output_size,
            mode='bicubic',
            align_corners=False
        ).squeeze()

        # Resize the interpolated depth map to match the input image dimensions
        resized_depth = interpolated_depth.cpu().numpy()

    #output_norm = cv2.normalize(resized_depth, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    output_norm = cv2.flip(output_norm, 1)
    frame = cv2.flip(frame, 1)

    cv2.imshow('Output', output_norm)
    cv2.imshow('Input', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
