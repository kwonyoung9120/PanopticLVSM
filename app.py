"""Gradio demo. Run locally with `python app.py`, or deploy as a Hugging Face Space."""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import PanopticLVSM

EXAMPLES = Path(__file__).resolve().parent / "docs" / "examples"
SCENES = ("scene0011_00", "scene0153_00", "scene0329_01", "scene0578_02", "scene0651_02")

model = PanopticLVSM(
    os.environ.get("CONFIG", "configs/demo.yaml"),
    os.environ.get("OVERRIDES", "").split(),
)


def write_video(frames, fps=20):
    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    return path


def run(view1, view2, seg1, seg2, cameras, n_frames):
    if None in (view1, view2, seg1, seg2, cameras):
        raise gr.Error("Provide both views, both segmentations, and the camera file.")
    out = model([view1, view2], [seg1, seg2], cameras, n_frames=int(n_frames))
    return write_video(out["rgb"]), write_video(out["seg"])


with gr.Blocks(title="PanopticLVSM") as demo:
    gr.Markdown(
        "# PanopticLVSM\n"
        "A frozen LVSM renders the camera path between two posed views, and the same network "
        "carries their segmentation along it.\n\n"
        "- **Segmentations**: one color per segment, the same color for the same segment in both "
        "views, black for unlabeled.\n"
        "- **Cameras**: JSON with `c2w` (two 4x4 OpenCV camera-to-world matrices) and "
        "`fxfycxcy` (two `[fx, fy, cx, cy]` in pixels at 256x256). Images are resized to 256x256."
    )
    with gr.Row():
        view1 = gr.Image(label="Input view 1", type="pil")
        view2 = gr.Image(label="Input view 2", type="pil")
        seg1 = gr.Image(label="Segmentation 1", type="pil", image_mode="RGB")
        seg2 = gr.Image(label="Segmentation 2", type="pil", image_mode="RGB")
    with gr.Row():
        cameras = gr.File(label="Cameras (JSON)", file_types=[".json"], type="filepath")
        n_frames = gr.Slider(20, 120, value=60, step=10, label="Frames per transition")
    button = gr.Button("Run", variant="primary")
    with gr.Row():
        rgb_video = gr.Video(label="Rendered RGB", autoplay=True, loop=True)
        seg_video = gr.Video(label="Rendered segmentation", autoplay=True, loop=True)
    button.click(run, [view1, view2, seg1, seg2, cameras, n_frames], [rgb_video, seg_video])
    gr.Examples(
        [
            [str(EXAMPLES / f"{scene}_{name}") for name in
             ("view1.jpg", "view2.jpg", "view1_seg.png", "view2_seg.png", "cameras.json")]
            for scene in SCENES
        ],
        [view1, view2, seg1, seg2, cameras],
        label="ScanNet examples (ground-truth segmentation and cameras)",
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
