"""
Road Damage Object Detection — Gradio Deployment App
Model: YOLO11s trained on RDD2022 China Drone subset

Run locally with:
    pip install -r requirements.txt
    python app.py
"""

import gradio as gr
from ultralytics import YOLO
from PIL import Image

# --- Load the best-performing model (YOLO11s) ---
MODEL_PATH = "weights/best_yolo11s.pt"
model = YOLO(MODEL_PATH)

CLASS_DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
    "Repair": "Previously Repaired Area",
}


def detect_road_damage(image, confidence_threshold):
    """Run YOLO11s inference on the uploaded image and return annotated image + report."""
    if image is None:
        return None, "Please upload an image first."

    results = model.predict(
        source=image,
        conf=confidence_threshold,
        imgsz=640,
        verbose=False,
    )
    result = results[0]

    # result.plot() returns a BGR numpy array; convert to RGB for display
    annotated_frame = result.plot()
    annotated_image = Image.fromarray(annotated_frame[..., ::-1])

    boxes = result.boxes
    if len(boxes) == 0:
        summary = "✅ No damage detected in this image at the current confidence threshold."
    else:
        class_counts = {}
        details = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            label = CLASS_DESCRIPTIONS.get(cls_name, cls_name)
            details.append(f"  • {cls_name} ({label}) — confidence: {conf:.1%}")

        summary = f"🔍 Detected {len(boxes)} damage instance(s):\n\n"
        summary += "\n".join(
            f"- {cls} ({CLASS_DESCRIPTIONS.get(cls, cls)}): {count}"
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])
        )
        summary += "\n\nDetails:\n" + "\n".join(details)

    return annotated_image, summary


with gr.Blocks(title="Road Damage Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🛣️ Road Damage Object Detection
        ### Powered by YOLO11s — trained on RDD2022 (China Drone subset)

        Upload a **drone / aerial-view** road image to detect surface damage.
        Supported classes: **D00** (longitudinal crack) • **D10** (transverse crack) •
        **D20** (alligator crack) • **D40** (pothole) • **Repair** (previously repaired area)

        > ⚠️ Note: this model was trained on aerial (drone) imagery only, and does not
        > generalize to ground-level photos taken from a phone or car camera.
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload road image")
            confidence_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                label="Confidence Threshold",
            )
            detect_btn = gr.Button("🔍 Detect Damage", variant="primary")

        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")
            output_text = gr.Textbox(label="Detailed Report", lines=10)

    detect_btn.click(
        fn=detect_road_damage,
        inputs=[input_image, confidence_slider],
        outputs=[output_image, output_text],
    )

    gr.Markdown(
        """
        ---
        **Model:** YOLO11s | **Dataset:** RDD2022 — China Drone Subset (2,393 images)
        """
    )

if __name__ == "__main__":
    demo.launch()
