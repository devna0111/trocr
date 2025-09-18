import os

import gradio as gr
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCRInferencer:
    def __init__(self):
        print("[INFO] Initialize TrOCR Inferencer.")
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

    def inference(self, image: Image) -> str:
        """Inference using model.

        It is performed as a procedure of preprocessing - inference - postprocessing.
        """
        # preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        # inference
        generated_ids = self.model.generate(pixel_values)
        # postprocess
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text

inferencer = TrOCRInferencer()


# Implement event function
def image_to_text(image: np.ndarray) -> str:
    image = Image.fromarray(image).convert("RGB")
    text = inferencer.inference(image)
    return text


# Implement app
with gr.Blocks() as app:
    gr.Markdown("# Handwritten Image OCR")
    with gr.Tab("Image upload"):
        image = gr.Image(label="Handwritten image file")
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=image, outputs=output
        )

        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[
                os.path.join(os.getcwd(), "examples/Hello.png"),
                os.path.join(os.getcwd(), "examples/Hello_cursive.png"),
                os.path.join(os.getcwd(), "examples/Red.png"),
                os.path.join(os.getcwd(), "examples/sentence.png"),
                os.path.join(os.getcwd(), "examples/i_love_you.png"),
                os.path.join(os.getcwd(), "examples/merrychristmas.png"),
                os.path.join(os.getcwd(), "examples/Rock.png"),
                os.path.join(os.getcwd(), "examples/Bob.png"),
            ],
            inputs=image,
            outputs=output,
            fn=image_to_text,
        )

    with gr.Tab("Drawing"):
        sketchpad = gr.Sketchpad(
            label="Handwritten Sketchpad",
            shape=(600, 192),
            brush_radius=2,
            invert_colors=False,
        )
        output = gr.Textbox(label="Output Box")
        convert_btn = gr.Button("Convert")
        convert_btn.click(
            fn=image_to_text, inputs=sketchpad, outputs=output
        )

app.launch(inline=False, share=True)

app.close()