import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Inferência de modelo de segmentação de vegetação."
    )
    parser.add_argument("--rgb", required=True, type=str)
    parser.add_argument("--modelpath", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    return parser.parse_args()


def preprocess_image(image_path: Path, image_size=(256, 256)):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(image_size)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_np = np.expand_dims(img_np, axis=0)
    return img, img_np


def postprocess_mask(prediction: np.ndarray, original_size):
    mask = prediction[0, :, :, 0]
    mask_binary = (mask > 0.5).astype(np.uint8)
    mask_visual = mask_binary * 255
    mask_img = Image.fromarray(mask_visual, mode="L")
    mask_img = mask_img.resize(original_size)
    return mask_img


def run_inference(rgb_path: Path, model_path: Path, output_path: Path):
    if not rgb_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {rgb_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    model = tf.keras.models.load_model(model_path)
    original_img, input_tensor = preprocess_image(rgb_path)
    prediction = model.predict(input_tensor)
    mask_img = postprocess_mask(prediction, original_img.size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(output_path)


def main():
    args = parse_arguments()

    run_inference(
        rgb_path=Path(args.rgb),
        model_path=Path(args.modelpath),
        output_path=Path(args.output)
    )


if __name__ == "__main__":
    main()
