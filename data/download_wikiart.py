"""
Script to download and organize WikiArt dataset.

WikiArt dataset can be obtained via:
1. Kaggle: https://www.kaggle.com/datasets/ipythonx/wikiart-gangogh-creating-art-gan
2. HuggingFace: huggingface.co/datasets/huggan/wikiart
3. Official scraping with permission

Usage:
    python data/download_wikiart.py --output_dir ./data/wikiart --source huggingface
    python data/download_wikiart.py --output_dir ./data/wikiart --source kaggle
    python data/download_wikiart.py --output_dir ./data/wikiart --verify
"""

import os
import argparse
import sys
from pathlib import Path


# 5 representative styles (subset for Colab disk constraints)
# Covers: abstract, impressionist, baroque, realism, cubism — enough for all experiments
WIKIART_STYLES = [
    "Abstract_Expressionism",
    "Impressionism",
    "Baroque",
    "Realism",
    "Cubism",
]

# Full 27 styles (use when disk space is not a concern, e.g. Oscar HPC)
WIKIART_STYLES_FULL = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism",
    "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Contemporary_Realism", "Cubism", "Early_Renaissance",
    "Expressionism", "Fauvism", "High_Renaissance", "Impressionism",
    "Mannerism_Late_Renaissance", "Minimalism", "Naive_Art_Primitivism",
    "New_Realism", "Northern_Renaissance", "Pointillism", "Pop_Art",
    "Post_Impressionism", "Realism", "Rococo", "Romanticism",
    "Symbolism", "Synthetic_Cubism", "Ukiyo_e",
]


def download_from_huggingface(output_dir: str):
    """
    Download WikiArt dataset from HuggingFace using the datasets library.
    Organizes images into style subdirectories.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Please install: pip install datasets --break-system-packages")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading WikiArt from HuggingFace (huggan/wikiart)...")
    print("This may take a while (~10GB)...")

    dataset = load_dataset("huggan/wikiart", split="train")

    # Get style label names
    style_names = dataset.features["style"].names
    print(f"Filtering to {len(WIKIART_STYLES)} styles: {WIKIART_STYLES}")

    # Save images per style (subset only)
    from tqdm import tqdm
    count = 0
    for idx, example in enumerate(tqdm(dataset, desc="Saving images")):
        style = style_names[example["style"]]
        if style not in WIKIART_STYLES:
            continue
        style_dir = output_path / style
        style_dir.mkdir(exist_ok=True)

        img_path = style_dir / f"{idx:06d}.jpg"
        if not img_path.exists():
            example["image"].save(img_path, "JPEG", quality=85)
        count += 1
    print(f"Saved {count} images across {len(WIKIART_STYLES)} styles.")

    print(f"\nDataset saved to {output_dir}")
    verify_dataset(output_dir)


def download_from_kaggle(output_dir: str):
    """
    Download WikiArt dataset from Kaggle.
    Requires kaggle API credentials in ~/.kaggle/kaggle.json
    """
    try:
        import kaggle
    except ImportError:
        print("Please install: pip install kaggle --break-system-packages")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading from Kaggle (requires API credentials)...")
    print("Ensure ~/.kaggle/kaggle.json exists with your credentials.")

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "ipythonx/wikiart-gangogh-creating-art-gan",
        path=str(output_path),
        unzip=True
    )

    print(f"Dataset downloaded to {output_dir}")
    verify_dataset(output_dir)


def verify_dataset(data_dir: str):
    """Check that the dataset is properly organized with style subdirectories."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: {data_dir} does not exist.")
        return False

    style_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not style_dirs:
        print(f"ERROR: No style subdirectories found in {data_dir}")
        return False

    total_images = 0
    print(f"\nDataset summary ({data_dir}):")
    print(f"{'Style':<40} {'Images':>8}")
    print("-" * 50)
    for style_dir in sorted(style_dirs):
        count = len(list(style_dir.glob("*.jpg"))) + len(list(style_dir.glob("*.png")))
        total_images += count
        print(f"{style_dir.name:<40} {count:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<40} {total_images:>8}")
    print(f"\nFound {len(style_dirs)} styles and {total_images} total images.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download WikiArt dataset")
    parser.add_argument("--output_dir", type=str, default="./data/wikiart",
                        help="Directory to save the dataset")
    parser.add_argument("--source", type=str, default="huggingface",
                        choices=["huggingface", "kaggle"],
                        help="Download source")
    parser.add_argument("--verify", action="store_true",
                        help="Only verify an existing dataset without downloading")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
    elif args.source == "huggingface":
        download_from_huggingface(args.output_dir)
    elif args.source == "kaggle":
        download_from_kaggle(args.output_dir)


if __name__ == "__main__":
    main()
