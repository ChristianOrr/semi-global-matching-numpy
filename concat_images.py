"""
This code concatenates the images in the readme for easy displaying. 
"""

import cv2
import os
import numpy as np

def concatenate_images():
    # Define the directory containing the images
    directory = 'cones'
    
    # Define the specific order of images as provided in the context
    image_order = [
        'left_right_colour.png',
        'left_right_grey.png', 
        'left_right_blur.png',
        'left_right_census.png',
        'left_right_cost_volume.png',
        'left_right_cost_agg.png',
        'left_right_disp.png',
        'left_right_depth_map.png'
    ]
    
    # Read images in the specified order
    images = []
    for file in image_order:
        img_path = os.path.join(directory, file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not read image {file}")
        else:
            print(f"Warning: Image {file} not found")
    
    # Check if we have any images
    if not images:
        print("No images found in the directory")
        return
    
    # Add numbers to top-left corner of each image
    for i, img in enumerate(images):
        number = i + 1
        # Put text on image with larger, black font
        cv2.putText(img, str(number), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    
    # Get the maximum width and total height
    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images)
    
    # Create a blank canvas with appropriate dimensions
    canvas = np.zeros((total_height, max_width, 3), dtype=np.uint8)
    
    # Place each image on the canvas
    y_offset = 0
    for img in images:
        x_offset = (max_width - img.shape[1]) // 2  # Center horizontally
        canvas[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img
        y_offset += img.shape[0]
    
    # Save the concatenated image
    output_path = os.path.join(directory, 'concatenated_images.png')
    cv2.imwrite(output_path, canvas)
    
    # Print dimensions of the final image
    print(f"Final concatenated image dimensions: {canvas.shape[1]} width x {canvas.shape[0]} height pixels")
    print(f"Concatenated image saved to {output_path}")

if __name__ == "__main__":
    concatenate_images()
