from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "NEU-DET"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def count_images(folder):
    return [
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

def inspect_split(split_name):
    image_root = DATASET_ROOT / split_name / "images"

    print(f"\n{'=' * 50}")
    print(f"{split_name.upper()} DATASET")
    print(f"{'=' * 50}")

    if not image_root.exists():
        print(f"ERROR: Image folder not found: {image_root}")
        return

    images = count_images(image_root)

    print(f"Total images: {len(images)}")

    category_counts = Counter(
        image.parent.name for image in images
    )

    print("\nImages per category:")

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nSample image paths:")

    for image in images[:5]:
        print(f"  {image.relative_to(PROJECT_ROOT)}")

    print(f"\nCategories found: {len(category_counts)}")

def main():
    print("NEU STEEL DEFECT DATASET INSPECTION")
    print(f"Dataset location: {DATASET_ROOT}")

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset folder does not exist.")
        return

    inspect_split("train")
    inspect_split("validation")

if __name__ == "__main__":
    main()