# using_colab = False

# if using_colab:
#     import torch
#     import torchvision
#     print("PyTorch version:", torch.__version__)
#     print("Torchvision version:", torchvision.__version__)
#     print("CUDA is available:", torch.cuda.is_available())
#     import sys
#     !{sys.executable} -m pip install opencv-python matplotlib scikit-learn
#     !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/sam3.git'
import torch
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time
import pandas as pd

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    prepare_masks_for_visualization,
    visualize_formatted_frame_output,
)

from huggingface_hub import login

# --- basic setup ---
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision("high")


sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
gpus_to_use = range(torch.cuda.device_count())
print(f"Using GPUs: {list(gpus_to_use)}")
predictor = build_sam3_video_predictor()

video_path = r"C:\Users\Milou\Documents\Project\S7\Sam3test\assets\videos\video_6frames.mp4"
# video_path = r'C:\Users\Milou\Documents\Project\S7\Sam3test\assets\videos\heroes_1080p_10s_15fps_100sec.mp4'

# --- read video frames ---
video_frames_for_vis = []
if video_path.endswith(".mp4"):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    video_frames_for_vis = glob.glob(os.path.join(video_path, "*.jpg"))
    video_frames_for_vis.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

# --- start SAM3 session ---
response = predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
        return_frame_outputs=True,
        stream_every_frame=True,
    )
)
session_id = response["session_id"]

# --- add prompt on frame 0 ---
prompt_text_str = "basketball player on the basketball field."
predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=0,
        text=prompt_text_str,
    )
)

# --- propagate through video and collect outputs ---
outputs_per_frame = {}
for response in predictor.handle_stream_request(
    request=dict(type="propagate_in_video", session_id=session_id)
):
    frame_idx = response.get("frame_index")
    frame_out = response.get("outputs")
    if frame_idx is not None and frame_out is not None:
        outputs_per_frame[frame_idx] = frame_out

# --- prepare all outputs at once ---
outputs_per_frame = prepare_masks_for_visualization(outputs_per_frame)

# --- output folder ---
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

csv_data = []

plt.close("all")
for fi in range(len(video_frames_for_vis)):
    print(f"Processing frame {fi}...")
    
    fig = visualize_formatted_frame_output(
        fi,
        video_frames_for_vis,
        outputs_list=[outputs_per_frame],
        titles=[f"SAM 3 Dense Tracking - Frame {fi}"],
        figsize=(6, 4),
        csv_data=csv_data
    )
    
    # --- Exact dezelfde figuur opslaan als PNG ---
    # save_path = output_dir / f"frame_{fi:05d}.png"
    # plt.savefig(str(save_path), bbox_inches="tight", dpi=100)
    # print(f"Saved to {save_path}")
    # save_path = fr'C:\Users\pvano\Documents\Opleiding\Fontys\Jaar 4\S7 project basketbal\sam3\output\output_image_{fi}.png'
    # plt.savefig(str(save_path), bbox_inches="tight", dpi=100)

    # print(f"Saved to {save_path}")
    # # plt.show()
    plt.close(fig)

if csv_data:
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "tracking_data_Sam3.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved tracking data to {csv_path}")

print(f"Saved {len(video_frames_for_vis)} frames with tracking info to {output_dir}")

def save_video_from_output_frames(
    model_size,
    image_folder="output",
    output_directory_videos="output",
    logger=None
) -> None:
    """Create a video file from a sequence of output frames stored as images."""
    if logger:
        start_time = time.time()
    
    # Lijst de afbeeldingen op en zorg ervoor dat ze in de juiste volgorde staan
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('output_image_')[1].split('.')[0]))
    
    # Haal de grootte van de eerste afbeelding op
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = first_image.shape
    
    video_name = f'output_video_model_{model_size}_duration_{len(images)//15}s.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(
        os.path.join(output_directory_videos, video_name),
        fourcc, 15,
        (width, height))
    
    # Loop door alle afbeeldingen en voeg ze toe aan de video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    # Maak de video object vrij
    video.release()
    print(f"Video created: {video_name}")
    
    if logger:
        end_time = time.time()
        logger.info(f'creating video completed in {end_time - start_time:.2f} seconds')

# --- Maak de video ---
save_video_from_output_frames('SAM3')