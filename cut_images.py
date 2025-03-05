import os
import json
from PIL import Image


categories = []
images = []

def crop_images(annotation_file, image_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the annotation file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Initialize the annotation list for cropped images
    cropped_annotations = []

    categories = annotations['categories']
    images = annotations['images']
    
    # Iterate over each annotation
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        image_name = images[image_id]['file_name']
        if images[image_id]['id'] != image_id:
            print(f"Image ID {image_id} does not match the image ID in the image file name {image_name}")
            exit(1)
        
        # Open the corresponding image file
        image_path = os.path.join(image_folder, f'{image_name}')
        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image file {image_path} not found")
            break
        
        # Get the category name
        category_name = None
        for category in categories:
            if category['id'] == category_id:
                category_name = category['name']
                break

        # Define the bounding box coordinates
        left, upper, width, height = bbox
        right = left + width
        lower = upper + height
        
        # Crop the image using the bounding box
        cropped_image = image.crop((left, upper, right, lower))
        
        # Save the cropped image to the output folder
        cropped_image_path = os.path.join(output_folder, f'cropped_image_{annotation["id"]}.jpg')
        cropped_image.save(cropped_image_path)
        
        # Add the cropped image annotation to the list
        cropped_annotations.append({
            'image_id': annotation['id'],
            'category_id': category_id,
            'category_name': category_name,
            'original_image': image_name,
            'file_name': f'cropped_image_{annotation["id"]}.jpg'
        })
    
    # Save the cropped annotations to a JSON file
    with open(os.path.join(output_folder, 'cropped_annotations.json'), 'w') as f:
        json.dump(cropped_annotations, f, indent=4)

# Example usage
annotation_file = 'annotations.json'
image_folder = 'images'
output_folder = 'cropped_images'
crop_images(annotation_file, image_folder, output_folder)