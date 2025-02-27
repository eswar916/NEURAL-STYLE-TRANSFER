# NEURAL-STYLE-TRANSFER
COMPANY: CODTECH IT SOLUTIONS

NAME: ALAHARI ESWAR CHANDRA VIDYA SAGAR

INTERN ID: CT12SBA

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 8 WEEKS

MENTOR: NEELA SANTOSH

# DESCRIPTION
The code performs Neural Style Transfer using a pre-trained model from TensorFlow Hub. This technique combines a content image and a style image to generate a new image that retains the content of the original image but adopts the artistic style of the second image.

EDITOR PLATFORM: VS Code

Neural Style Transfer: The task is to apply the visual style of one image (the style image) to another image (the content image). The result is a "styled" version of the content image.

Key Steps in the Code: Loading and Preprocessing: The images are loaded from file paths, resized to 256x256 pixels (for efficiency), and normalized to a range of [0, 1] to match the input requirements of the model.

Displaying Images: The display_images function shows the original content image, style image, and the generated image side by side for comparison.

Using Pre-trained Model: The model used for style transfer is loaded from TensorFlow Hub: https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1. This model applies the style of the style image onto the content image.

Generating and Saving Styled Image: The output is the stylized_image, which is the result of applying style transfer. The result is then saved to the user-provided output path.

Applicable Use Cases: Artistic Image Generation:Turn photos into artwork by applying the style of famous painters like Van Gogh, Picasso, etc., to your own images. Digital Content Creation:Content creators can use style transfer to create visually unique images for websites, blogs, social media posts, and other digital content. Graphic Design:This could be used in graphic design for adding artistic styles to logos, promotional images, or other media elements. Fashion and Design:Designers could experiment with patterns and color palettes by using style transfer to visualize different design concepts on clothing, interiors, or other designs. Photography:Photographers can add artistic effects to their photographs, creating visually appealing and stylized versions of their work. Interactive Art:This could be used in interactive installations or applications where users can apply artistic styles to images in real-time.

Dependencies: The following libraries are required: tensorflow: The main library used for building and deploying machine learning models. tensorflow_hub: This allows loading pre-trained models hosted on TensorFlow Hub. numpy: Used for handling arrays and numerical computations. matplotlib: Used for displaying images. PIL (Pillow): Used for image manipulation (loading, saving, etc.).

# OUTPUT
