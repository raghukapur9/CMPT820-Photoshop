import struct
import heapq
import numpy as np

def parse_bmp(filename):
    with open(filename, 'rb') as f:

        file_header = f.read(14)
        file_type, file_size, _, _ , data_offset = struct.unpack('<2sI2HI', file_header)

        if file_type != b'BM':
            raise ValueError('File is not a valid BMP file')

        info_header = f.read(40)
        _, width, height, _, bits_per_pixel, compression, image_size, _, _, _, _ = struct.unpack('<IIIHHIIIIII', info_header)

        # Validate the BMP format
        if bits_per_pixel != 24:
            raise ValueError('Only 24-bit BMP files are supported')
        if compression != 0:
            raise ValueError('Only uncompressed BMP files are supported')

        if image_size == 0:
            image_size = file_size - data_offset

        # Move to the start of the pixel data
        f.seek(data_offset)

        image_data = f.read(image_size)

        return {
            'width': width,
            'height': height,
            'data': image_data
        }

def get_grayscale_values(np_image):
    gray_values = 0.299 * np_image[:, :, 0] + 0.587 * np_image[:, :, 1] + 0.114 * np_image[:, :, 2]
    return gray_values

def build_huffman_tree(image_histogram):
    heap = [[weight, [symbol, ""]] for symbol, weight in image_histogram.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return heap[0]

def calculate_average_code_length(huffman_tree, image_histogram):
    codes = {}
    def encode_huffman_codes(tree_node, prefix=""):
        if len(tree_node) == 2:
            codes[tree_node[0]] = prefix
        else:
            encode_huffman_codes(tree_node[1], prefix + "0")
            encode_huffman_codes(tree_node[2], prefix + "1")

    encode_huffman_codes(huffman_tree)
    total_pixels = sum(image_histogram.values())
    average_code_length = sum(
        len(code) * (freq / total_pixels) for symbol, code in codes.items() for symbol, freq in image_histogram.items() if symbol == symbol
    )
    return average_code_length

def get_color_component(image, index):
    row,col,plane = image.shape
    component = np.zeros((row,col,plane),np.uint8)
    component[:,:,index] = image[:,:,index]

    return component

def invert_image(image):
    return 255 - image

def gaussian_kernel(size, sigma=1.0):
    if sigma <= 0:
        sigma = 1.0
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def gaussian_blur(image, kernel_size, sigma=0):
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_width = kernel_size // 2
    if len(image.shape) == 2:  # Grayscale image
        padded_image = np.pad(image, pad_width, mode='edge')
        blurred_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                blurred_image[i, j] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])
    else:  # Color image
        blurred_image = np.zeros_like(image)
        for c in range(image.shape[2]):  # Apply the blur to each channel
            padded_image = np.pad(image[:, :, c], pad_width, mode='edge')
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    blurred_image[i, j, c] = np.sum(kernel * padded_image[i:i+kernel_size, j:j+kernel_size])
    
    return blurred_image

def divide_images(image1, image2, scale=255):
    result = (image1 / np.clip(image2, 1, 255)) * scale
    return np.clip(result, 0, 255).astype(np.uint8)

def get_sepia_values(np_image):
    # Apply a sepia tone
    tr = 0.393 * np_image[:,:,0] + 0.769 * np_image[:,:,1] + 0.189 * np_image[:,:,2]
    tg = 0.349 * np_image[:,:,0] + 0.686 * np_image[:,:,1] + 0.168 * np_image[:,:,2]
    tb = 0.272 * np_image[:,:,0] + 0.534 * np_image[:,:,1] + 0.131 * np_image[:,:,2]

    sepia_image = np.stack([tr, tg, tb], axis=2)
    
    return sepia_image

def add_gaussian_noise_to_image(np_image, mean=0, std=25):
    noise = np.random.normal(mean, std, np_image.shape)

    np_noisy_image = np_image + noise

    np_noisy_image = np.clip(np_noisy_image, 0, 255)
    return np_noisy_image