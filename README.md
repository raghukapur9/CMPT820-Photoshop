# CMPT 820 Multimedia Systems Mini-Photoshop

Project is built in Python using the Tkinter library for building the GUI, along with libraries such as openCV, PIL, numpy.

## Steps to run the project
- Clone the project ```git clone https://github.com/raghukapur9/CMPT820-Photoshop.git```
- Install python and create a virtual enviornment.
- ```pip3 install -r requirements.txt```
- ```python3 mini_photoshop.py```

## Operations included

### Core Operations
    - Open a file
    - Exit
    - Grayscale Conversion
    - Ordered Dithering
    - Auto Level
    - Huffman Coding

### Optional Operations
    - Image Negative Conversion
    - Red Channel Display
    - Green Channel Display
    - Blue Channel Display
    - Edge Detection
    - Pencil Sketch Effect
    - Color Pencil Sketch Effect
    - Cartoon Effect
    - Background Removal

## Detail of each operation
- **Open a file**: Opens a file dialog box for the user to select a BMP file. Once selected, the program will then display the BMP file.
- **Exit**: This will terminate the program
- **Grayscale**: This function converts an image to grayscale and displays alongside the original image. It uses the luminosity formula for conversion and provides feedback if no image is loaded.
- **Ordered Dithering**: This function applies ordered dithering to an image using a 4x4 Bayer matrix, converting it to a dithered grayscale image. Grayscale image is shown on the left side whereas the ordered dithered image is shown on the right. It first ensures an image is loaded.
- **Auto Level**: This function applies an auto-leveling process to an image by adjusting its histogram to stretch across the full range of pixel values [0, 255], and then displays the auto-leveled image along with the original one. It first checks that the original image is loaded.
- **Huffman Coding**: This function calculates and displays the entropy and average Huffman code length for an image, offering insights into its compressibility and efficiency of the Huffman coding. It ensures an image is loaded, computes its grayscale histogram, and uses this for Huffman tree construction and statistical calculations.
- **Image Negative Conversion**: This function converts an image to its negative by inverting the pixel values and displays the result.
- **Red Channel Display**: This function isolates and displays the red color channel of an image, ensuring an image is loaded before proceeding with the extraction.
- **Green Channel Display**: This function isolates and displays the green color channel of an image, ensuring an image is loaded before proceeding with the extraction.
- **Blue Channel Display**: This function isolates and displays the blue color channel of an image, ensuring an image is loaded before proceeding with the extraction.
- **Edge Detection**: This function converts an image to grayscale, applies binary thresholding, then uses the Canny edge detection algorithm to identify and display the edges.
- **Pencil Sketch Effect**: This function applies a pencil sketch effect to an image by inverting the grayscale version, applying Gaussian blur, and then blending it with the original grayscale image, displaying the result.
- **Color Pencil Sketch Effect**: This function applies a color pencil sketch effect to an image by inverting it, applying Gaussian blur, then blending it back with the original, displaying the enhanced image.
- **Cartoon Effect**: This function transforms an image into a cartoon-like effect by first converting it to grayscale, applying median blur for smoothing, and creating an edge mask using adaptive thresholding. It then applies a bilateral filter to the original image to reduce color palette while preserving edges and combines this with the edge mask to produce the cartoon effect.
- **Background Removal**: This function implements background removal from an image by first allowing the user to select a region of interest through cv2.selectROI, then applying the GrabCut algorithm with the selected region to differentiate foreground from background. The result is a modified image where the background is removed.

