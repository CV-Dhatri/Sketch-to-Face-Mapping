import cv2
import numpy as np
import os

def generate_sketch_inverse_gct(image_path, output_path):
    # 1. Image Retrieval
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find image at {image_path}")
        return
    
    img = img.astype(np.float32)

    # 2. Color Elimination Stage
    grayscale = np.mean(img, axis=2)

    # 3. Transformer Stage (Inverse Transform)
    inverse = 255.0 - grayscale

    # 4. Mean Transform
    mean_filtered = cv2.blur(inverse, (3, 3))

    # 5. Merging of Grayscale and Transformed Image
    # Divide the matrices and multiply by 255. 
    # Normal background pixels (1.0) become 255 (White).
    merged = (inverse / (mean_filtered + 1e-5)) * 255.0
    
    # Clip the values to strictly stay within the 0-255 image byte range
    merged_clipped = np.clip(merged, 0, 255)

    # 6. Gamma Adjustment Stage
    # The paper explicitly states we must darken the image to maintain a hand-drawn appearance.
    # We map to [0,1], raise to the power of 2.2 (which darkens the values), and map back to [0,255].
    gamma = 2.2
    sketch = np.power(merged_clipped / 255.0, gamma) * 255.0
    final_sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    # Save the output
    cv2.imwrite(output_path, final_sketch)
    print(f"Success! Sketch saved to {output_path}")

import glob

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, '..', 'data', 'photos')
    output_dir = os.path.join(script_dir, '..', 'data', 'processed_sketches')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Grab all jpg files in the input directory
    image_files = glob.glob(os.path.join(input_dir, '*.jpg'))
    
    print(f"Found {len(image_files)} images. Starting batch processing...")
    
    success_count = 0
    for img_path in image_files:
        # Extract just the filename (e.g., 'f-005-01.jpg')
        filename = os.path.basename(img_path)
        
        # We append '_gct' to distinguish them, though you can keep the same name
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_gct{ext}"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            generate_sketch_inverse_gct(img_path, output_path)
            success_count += 1
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    print(f"\nBatch processing complete! Successfully generated {success_count} sketches.")