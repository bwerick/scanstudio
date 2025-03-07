import os
import numpy as np
from sam2 import SAM2Model  # Assuming SAM2Model is the model class from the SAM2 library
from PIL import Image

def load_images(image_dir, batch_size=8):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    return batches

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256, 256))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0  # Normalize the image
    return image

def postprocess_result(result):
    # Assuming result is a numpy array with segmentation masks
    return (result * 255).astype(np.uint8)

def save_result(result, output_path):
    result_image = Image.fromarray(result)
    result_image.save(output_path)

def infer_batch(model, image_batch):
    preprocessed_images = [preprocess_image(img_path) for img_path in image_batch]
    preprocessed_images = np.stack(preprocessed_images, axis=0)
    results = model.predict(preprocessed_images)
    return results

def main(image_dir, output_dir):
    model = SAM2Model()  # Initialize the model
    model.load_weights('path/to/weights')  # Load pre-trained weights

    image_batches = load_images(image_dir)
    for batch_idx, image_batch in enumerate(image_batches):
        results = infer_batch(model, image_batch)
        for img_idx, result in enumerate(results):
            output_path = os.path.join(output_dir, f'result_{batch_idx * 8 + img_idx}.png')
            postprocessed_result = postprocess_result(result)
            save_result(postprocessed_result, output_path)

if __name__ == "__main__":
    image_dir = '/path/to/images'
    output_dir = '/path/to/output'
    os.makedirs(output_dir, exist_ok=True)
    main(image_dir, output_dir)