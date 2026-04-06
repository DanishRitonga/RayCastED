r"""RayCastED — Gradio Demo Interface.

Interactive web demo for polygon cell detection. Upload a WSI tile,
adjust the confidence threshold, and see detected polygon boundaries
overlaid on the image.

Usage:
    uv run python -m raycasted.deploy.gradio.app \
        --weights runs/detect/train/weights/best.pt

    # With public sharing link:
    uv run python -m raycasted.deploy.gradio.app \
        --weights best.pt --share
"""

import argparse

import gradio as gr
import numpy as np

from raycasted.deploy.gradio.inference import RayCastDemoInference


def create_demo(weights_path: str, device: str = 'auto') -> gr.Blocks:
    """Build the Gradio interface for RayCastED polygon detection.

    Args:
        weights_path: Path to trained .pt checkpoint.
        device: 'auto', 'cpu', or 'cuda'.

    Returns:
        Gradio Blocks app ready to launch.
    """
    inference = RayCastDemoInference(weights_path, device)

    with gr.Blocks(title='RayCastED — Polygon Cell Detector') as demo:
        gr.Markdown('# RayCastED — Polygon Cell Detector')
        gr.Markdown(
            'Upload a histopathological tissue image (WSI tile) to detect '
            'individual cells as polygon boundaries using 32-ray star-convex shapes.'
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type='numpy', label='Input Image')
                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label='Confidence Threshold',
                    )
                    show_rays = gr.Checkbox(label='Show Rays', value=False)
                run_btn = gr.Button('Detect Cells', variant='primary')

            with gr.Column(scale=1):
                image_output = gr.Image(type='numpy', label='Detections')
                stats_output = gr.Textbox(label='Statistics', lines=4)

        def _predict(image: np.ndarray | None, conf: float, rays: bool):
            if image is None:
                return None, 'Upload an image to start.'
            return inference.predict(image, conf_threshold=conf, show_rays=rays)

        run_btn.click(
            fn=_predict,
            inputs=[image_input, conf_slider, show_rays],
            outputs=[image_output, stats_output],
        )

    return demo


def main():
    """CLI entry point for the Gradio demo."""
    parser = argparse.ArgumentParser(description='RayCastED Gradio Demo')
    parser.add_argument('--weights', required=True, help='Path to trained .pt checkpoint')
    parser.add_argument('--device', default='auto', help='Device: auto, cpu, cuda')
    parser.add_argument('--share', action='store_true', help='Create a public sharing link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run on')
    args = parser.parse_args()

    demo = create_demo(args.weights, args.device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
