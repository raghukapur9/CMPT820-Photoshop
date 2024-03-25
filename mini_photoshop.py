import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from collections import Counter
# from scipy.stats import entropy
import heapq

class MiniPhotoshop:
    def __init__(self, root):
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.root.title("Mini Photoshop")

        # Menu setup
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # File menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open File", command=self.open_file)
        self.file_menu.add_command(label="Grayscale", command=self.convert_to_grayscale)
        self.file_menu.add_command(label="Ordered Dithering", command=self.apply_ordered_dithering)
        self.file_menu.add_command(label="Auto Level", command=self.apply_auto_level)
        self.file_menu.add_command(label="Huffman", command=self.show_huffman_stats)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.exit_app)
        self.menu_bar.add_cascade(label="Core Operations", menu=self.file_menu)

        # Image labels for original and grayscale images
        self.original_image_label = tk.Label(self.root)
        self.original_image_label.pack(side=tk.LEFT)

        self.grayscale_image_label = tk.Label(self.root)
        self.grayscale_image_label.pack(side=tk.RIGHT)

        self.current_image = None

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Bitmap files", "*.bmp")])
        if file_path:
            try:
                self.current_image = Image.open(file_path)
                self.display_image(self.current_image, self.original_image_label)
                # Clear the grayscale label
                self.grayscale_image_label.config(image='')
                self.grayscale_image_label.image = None
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open the file: {e}")

    def display_image(self, image, label):
        image.thumbnail((1024, 768))  # Resize to fit display area
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def convert_to_grayscale(self):
        if self.current_image:
            gray_image = self.current_image.convert("L")  # Convert image to grayscale
            self.display_image(gray_image, self.grayscale_image_label)
        else:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")

    def apply_ordered_dithering(self):
        if not self.current_image:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        
        # Convert image to grayscale first
        gray_image = self.current_image.convert("L")
        self.display_image(gray_image, self.original_image_label)

        # Define a 4x4 Bayer matrix
        bayer_matrix = np.array([[0, 8, 2, 10],
                                [12, 4, 14, 6],
                                [3, 11, 1, 9],
                                [15, 7, 13, 5]]) / 15.0 * 255.0

        # Scale the Bayer matrix to the size of the image
        rows, cols = gray_image.size
        repeated_matrix = np.tile(bayer_matrix, (rows // 4 + 1, cols // 4 + 1))
        repeated_matrix = repeated_matrix[:rows, :cols]

        # Convert PIL Image to NumPy array for manipulation
        img_array = np.array(gray_image)

        # Apply ordered dithering
        dithered_img_array = img_array + (repeated_matrix - 128)
        dithered_img_array = np.clip(dithered_img_array, 0, 255)

        # Threshold the image
        dithered_img_array = np.where(dithered_img_array < 128, 0, 255)

        # Convert back to PIL Image
        dithered_image = Image.fromarray(np.uint8(dithered_img_array))

        # Display the dithered image
        self.display_image(dithered_image, self.grayscale_image_label)

    def apply_auto_level(self):
        if not self.current_image:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        # Ensure we're working with a copy of the image in RGB
        image = self.current_image.convert("RGB")
        self.display_image(image, self.original_image_label)
        
        # Convert the image to a NumPy array for easier manipulation
        img_array = np.array(image)

        # Calculate the minimum and maximum pixel values
        minval = img_array.min(axis=(0, 1))
        maxval = img_array.max(axis=(0, 1))

        # Prevent division by zero and ensure there's a range
        maxval[maxval == minval] = 255
        minval[minval == maxval] = 0

        # Apply auto level: stretch the histogram to cover the full [0, 255] range
        auto_leveled_img = (img_array - minval) / (maxval - minval) * 255
        auto_leveled_img = np.clip(auto_leveled_img, 0, 255).astype(np.uint8)

        # Convert back to a PIL Image and display
        auto_leveled_image = Image.fromarray(auto_leveled_img)
        self.display_image(auto_leveled_image, self.grayscale_image_label)

    def build_huffman_tree(self, image_histogram):
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

    def calculate_average_code_length(self, huffman_tree, image_histogram):
        symbols = huffman_tree[1:]
        total_bits = sum([len(code) * image_histogram[int(symbol)] for _, symbol, code in self.flatten(symbols)])
        total_pixels = sum(image_histogram.values())
        return total_bits / total_pixels

    def flatten(self, lst):
        for item in lst:
            if isinstance(item, list) and len(item) == 2 and isinstance(item[0], int):
                yield item
            elif isinstance(item, list):
                yield from self.flatten(item)

    def show_huffman_stats(self):
        if not self.current_image:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        
        gray_image = self.current_image.convert("L")
        img_array = np.array(gray_image)
        histogram = Counter(img_array.flatten())
        total_pixels = sum(histogram.values())

        probabilities = {k: v / total_pixels for k, v in histogram.items()}
        img_entropy = -sum(p * np.log2(p) for p in probabilities.values())

        huffman_tree = self.build_huffman_tree(histogram)
        avg_code_length = self.calculate_average_code_length(huffman_tree, histogram)

        messagebox.showinfo("Huffman Stats", f"Entropy: {img_entropy:.2f} bits\nAverage Huffman Code Length: {avg_code_length:.2f} bits")

    def exit_app(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()
