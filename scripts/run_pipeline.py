import argparse
import subprocess
from pathlib import Path
import sys
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthomosaic", required=True, type=str)
    parser.add_argument("--inference_image", required=True, type=str)
    parser.add_argument("--tile_size", type=int, default=256)
    parser.add_argument("--modelpath", type=str, default="models/vegetation_model.h5")
    return parser.parse_args()


def run_command(cmd):
    print(f"\n[RUN] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR]")
        print(result.stderr)
        sys.exit(1)
    print(result.stdout)


def clean_directory(path: Path):
    if path.exists():
        print(f"[CLEAN] {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_arguments()

    project_root = Path(__file__).resolve().parent.parent

    orthomosaic = Path(args.orthomosaic).resolve()
    inference_image = Path(args.inference_image).resolve()
    model_path = (project_root / args.modelpath).resolve()

    tiles_dir = project_root / "data/tiles"
    masks_dir = project_root / "data/masks"
    inference_out_dir = project_root / "data/inference_outputs"
    output_inference = inference_out_dir / "segmented.png"

    if not orthomosaic.exists():
        print(f"[ERROR] {orthomosaic}")
        sys.exit(1)

    if not inference_image.exists():
        print(f"[ERROR] {inference_image}")
        sys.exit(1)

    clean_directory(tiles_dir)
    clean_directory(masks_dir)
    clean_directory(inference_out_dir)

    run_command([
        "python", "src/preprocessing/divide_orthomosaic.py",
        "--input", str(orthomosaic),
        "--output", str(tiles_dir),
        "--tile_size", str(args.tile_size)
    ])

    run_command([
        "python", "src/dataset/binarize_images.py",
        "--input", str(tiles_dir),
        "--output", str(masks_dir)
    ])

    run_command([
        "python", "scripts/validate_dataset.py",
        "--rgb", str(tiles_dir),
        "--masks", str(masks_dir)
    ])

    run_command([
        "python", "scripts/count_mask_values.py",
        "--masks", str(masks_dir)
    ])

    run_command([
        "python", "src/training/train_model.py",
        "--rgb", str(tiles_dir),
        "--groundtruth", str(masks_dir),
        "--modelpath", str(model_path)
    ])

    run_command([
        "python", "src/inference/model_inference.py",
        "--rgb", str(inference_image),
        "--modelpath", str(model_path),
        "--output", str(output_inference)
])


    print("\n[DONE]")
    print(output_inference.resolve())


if __name__ == "__main__":
    main()
