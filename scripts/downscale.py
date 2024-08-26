import argparse
from PIL import Image
from pathlib import Path

def downscale_images(input_dir: Path, output_dir: Path):
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all PNG images in the input directory
    for img_path in input_dir.glob("*.png"):
        # Open the image
        with Image.open(img_path) as img:
            # Calculate the new size (half of the original)
            new_size = (img.width // 2, img.height // 2)
            
            # Downscale the image
            img_downscaled = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save the downscaled image to the output directory
            output_path = output_dir / img_path.name
            img_downscaled.save(output_path)

    print(f"Downscaled images saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Downscale PNG images by 1/2 and save to a specified directory.")
    
    parser.add_argument("--input", "-i", type=Path, help="Path to the input directory containing PNG images.")
    parser.add_argument("--output", "-o", type=Path, help="Path to the output directory where downscaled images will be saved.")
    
    args = parser.parse_args()
    
    downscale_images(args.input, args.output)

if __name__ == "__main__":
    main()
