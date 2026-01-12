import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Valida dataset binário (0 = não vegetação, 1 = vegetação)."
    )
    parser.add_argument(
        "--rgb",
        required=True,
        type=str,
        help="Diretório contendo imagens RGB (blocos)."
    )
    parser.add_argument(
        "--masks",
        required=True,
        type=str,
        help="Diretório contendo máscaras binárias (0/1)."
    )
    return parser.parse_args()


def validate_pair(rgb_path: Path, mask_path: Path):
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path)

    rgb_np = np.array(rgb)
    mask_np = np.array(mask)

    errors = []

    # 1. Dimensões
    if rgb_np.shape[0] != mask_np.shape[0] or rgb_np.shape[1] != mask_np.shape[1]:
        errors.append("Dimensões não coincidem")

    # 2. Escala de cinza
    if len(mask_np.shape) != 2:
        errors.append("Máscara não está em escala de cinza (1 canal)")

    # 3. Valores binários
    unique_vals = np.unique(mask_np)
    if not set(unique_vals).issubset({0, 1}):
        errors.append(f"Valores inválidos na máscara: {unique_vals}")

    return errors


def validate_dataset(rgb_dir: Path, masks_dir: Path):
    if not rgb_dir.exists():
        raise FileNotFoundError(f"Diretório RGB não encontrado: {rgb_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Diretório de máscaras não encontrado: {masks_dir}")

    rgb_files = sorted([p for p in rgb_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    mask_files = sorted([p for p in masks_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    if not rgb_files:
        raise RuntimeError("Nenhuma imagem RGB encontrada.")
    if not mask_files:
        raise RuntimeError("Nenhuma máscara encontrada.")

    print(f"[INFO] Imagens RGB: {len(rgb_files)}")
    print(f"[INFO] Máscaras: {len(mask_files)}")

    # 1. Correspondência de arquivos
    rgb_names = {p.name for p in rgb_files}
    mask_names = {p.name for p in mask_files}

    missing_masks = rgb_names - mask_names
    extra_masks = mask_names - rgb_names

    if missing_masks:
        print(f"[ERROR] Faltam máscaras para {len(missing_masks)} imagens:")
        for name in list(missing_masks)[:5]:
            print(f"  - {name}")

    if extra_masks:
        print(f"[ERROR] Máscaras sem imagem correspondente: {len(extra_masks)}")
        for name in list(extra_masks)[:5]:
            print(f"  - {name}")

    if missing_masks or extra_masks:
        print("[FAIL] Correspondência entre imagens e máscaras inválida.")
        return

    # 2. Validação conteúdo
    total = len(rgb_files)
    error_count = 0

    for idx, rgb_path in enumerate(rgb_files):
        mask_path = masks_dir / rgb_path.name
        errors = validate_pair(rgb_path, mask_path)

        if errors:
            error_count += 1
            print(f"[ERROR] {rgb_path.name}: {', '.join(errors)}")

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[INFO] Verificadas {idx + 1}/{total} imagens...")

    # 3. Resultado final
    if error_count == 0:
        print(f"[DONE] Dataset válido. {total} pares verificados com sucesso.")
    else:
        print(f"[FAIL] Dataset inválido. {error_count} arquivos com erro.")


def main():
    args = parse_arguments()
    validate_dataset(
        rgb_dir=Path(args.rgb),
        masks_dir=Path(args.masks)
    )


if __name__ == "__main__":
    main()
