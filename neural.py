import os
import tensorflow as tf

try:
    import tensorflow_hub as hub
except ModuleNotFoundError:
    print("The 'tensorflow_hub' module is not installed. Install it using: pip install tensorflow-hub")
    exit()

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Disable TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")  # Suppress warnings and info logs

# Function to load and preprocess the image
def load_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize for faster processing
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = img.astype(np.float32)  # Ensure the image is float32
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to display images
def display_images(content_image, style_image, generated_image):
    plt.figure(figsize=(12, 12))

    plt.subplot(1, 3, 1)
    plt.title("Content Image")
    plt.imshow(content_image[0])  # Remove batch dimension for display
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Style Image")
    plt.imshow(style_image[0])  # Remove batch dimension for display
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Generated Image")
    plt.imshow(generated_image)  # No need to use [0] since batch dimension is removed
    plt.axis('off')

    plt.show()

# Main function for Fast Style Transfer
def neural_style_transfer(content_path, style_path):
    # Load images
    content_image = load_image(content_path)
    style_image = load_image(style_path)

    # Load the pre-trained model from TensorFlow Hub
    try:
        model = hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1")
    except Exception as e:
        print(f"Error loading TensorFlow Hub model: {e}")
        exit()

    # Perform style transfer
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

    # Convert the generated image to a displayable format
    stylized_image = stylized_image.numpy()  # Convert Tensor to NumPy array
    stylized_image = (stylized_image * 255).astype(np.uint8)  # Scale and convert to uint8
    stylized_image = np.squeeze(stylized_image, axis=0)  # Remove batch dimension

    return content_image, style_image, stylized_image

if __name__ == "__main__":
    # User input for content and style images
    content_path = input("Enter the path to the content image: ").strip()
    style_path = input("Enter the path to the style image: ").strip()

    if not os.path.exists(content_path) or not os.path.exists(style_path):
        print("Error: One or both image paths are invalid.")
        exit()

    # Run Fast Style Transfer
    content_image, style_image, generated_image = neural_style_transfer(content_path, style_path)

    # Display the results
    display_images(content_image, style_image, generated_image)

    # Save the generated image
    output_path = input("Enter the path to save the styled image (e.g., styled_image.jpg): ").strip()
    Image.fromarray(generated_image).save(output_path)
    print(f"Styled image saved at {output_path}")
