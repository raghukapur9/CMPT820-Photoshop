import tkinter as tk
from tkinter import filedialog, messagebox, Label, Button
from PIL import Image, ImageTk
import numpy as np
from collections import Counter
import cv2
from utils import parse_bmp, get_grayscale_values, build_huffman_tree, calculate_average_code_length, get_color_component, \
divide_images, gaussian_blur, invert_image, get_sepia_values, add_gaussian_noise_to_image

class MiniPhotoshop:

    def __init__(self, root):
        self.panelA = None
        self.panelB = None
        self.image = None
        self.root = root
        self.root.attributes('-fullscreen', True)
        self.root.title("Mini Photoshop CMPT 820")
        # Label for the submission note
        submitted_by_label = Label(root, text="Submitted By: Raghu Kapur",
            fg="black", bg="lightgray", borderwidth=5, font=('Arial', 15))
        submitted_by_label.grid(row=0, column=0, padx=2, pady=10, sticky='nesw')

        l1= Label(root, text="CLICK THE BUTTONS TO PERFORM THE FUNCTIONALITIES MENTIONED",
           fg="white", bg="blue", width= 98, borderwidth=5, font =('Arial', 15))

        l1.grid(row= 0, column= 1, columnspan= 6, padx=2, pady=10, sticky='nesw')

        l2 = Label(root, text="Core Operations",
           fg="white", bg="blue", borderwidth=5, font =('Arial', 15))
        l2.grid(row= 1, column= 0, padx=2, pady=10, sticky='nesw')

        btn= Button(root, text="OPEN FILE", fg="black", bg="lavender", command=self.open_file)
        btn.grid(row= 2, column= 0, padx=2, pady=10, sticky='nesw')

        btn1 = Button(root, text="EXIT", fg="black", bg="lavender", command=self.exit_app)
        btn1.grid(row= 3, column= 0, padx=2, pady=10, sticky='nesw')

        btn2 = Button(root, text="CONVERT TO GRAYSCALE", fg="black", bg="lavender", command=self.convert_to_grayscale)
        btn2.grid(row= 4, column= 0, padx=2, pady=10, sticky='nesw')

        btn3 = Button(root, text="ORDERED DITHERING", fg="black", bg="lavender", command=self.apply_ordered_dithering)
        btn3.grid(row= 5, column= 0, padx=2, pady=10, sticky='nesw')

        btn4 = Button(root, text="AUTO LEVEL", fg="black", bg="lavender", command=self.apply_auto_level)
        btn4.grid(row= 6, column= 0, padx=2, pady=10, sticky='nesw')

        btn5 = Button(root, text="HUFFMAN CODING", fg="black", bg="lavender", command=self.show_huffman_stats)
        btn5.grid(row= 7, column= 0, padx=2, pady=10, sticky='nesw')

        l3 = Label(root, text="Optional Operations",
           fg="white", bg="blue", borderwidth=5, font =('Arial', 15))
        l3.grid(row= 8, column= 0, padx=2, pady=10, sticky='nesw')

        btn6 = Button(root, text="NEGATIVE IMAGE", fg="black", bg="lavender", command=self.negative)
        btn6.grid(row= 9, column= 0, padx=2, pady=10, sticky='nesw')

        btn7 = Button(root, text="EXTRACT RED", fg="black", bg="lavender", command=self.extract_red)
        btn7.grid(row= 10, column= 0, padx=2, pady=10, sticky='nesw')

        btn8 = Button(root, text="EXTRACT BLUE", fg="black", bg="lavender", command=self.extract_blue)
        btn8.grid(row=11, column= 0, padx=2, pady=10, sticky='nesw')

        btn9 = Button(root, text="EXTRACT GREEN", fg="black", bg="lavender", command=self.extract_green)
        btn9.grid(row= 12, column= 0, padx=2, pady=10, sticky='nesw')

        btn10 = Button(root, text="EDGE DETECTION", fg="black", bg="lavender", command=self.detect_edges)
        btn10.grid(row= 13, column= 0, padx=2, pady=10, sticky='nesw')

        btn11 = Button(root, text="PENCIL SKETCH EFFECT", fg="black", bg="lavender", command=self.pencil_sketch_effect)
        btn11.grid(row= 14, column= 0, padx=2, pady=10, sticky='nesw')

        btn12 = Button(root, text="COLOR PENCIL SKETCH EFFECT", fg="black", bg="lavender", command=self.color_pencil_sketch_effect)
        btn12.grid(row= 15, column= 0, padx=2, pady=10, sticky='nesw')

        btn14 = Button(root, text="SEPIA FILTER", fg="black", bg="lavender", command=self.apply_sepia_filter)
        btn14.grid(row= 17, column= 0, padx=2, pady=10, sticky='nesw')

        btn15 = Button(root, text="GAUSSIAN NOISE", fg="black", bg="lavender", command=self.apply_gaussian_noise)
        btn15.grid(row= 18, column= 0, padx=2, pady=10, sticky='nesw')

    def open_file(self):
        f_types = [
            ("BMP", ".bmp"),
            ("JPEG", ".jpg"),
            ("JPEG", ".jpeg"),
            ("PNG", ".png")
        ]

        path = filedialog.askopenfilename(filetypes=f_types)

        if not path:
            return

        array = None
        if path.lower().endswith('.bmp'):
            image_data = parse_bmp(path)

            # Convert the bytes data to a NumPy array
            image_data = parse_bmp(path)

            # Convert the bytes data to a NumPy array
            height, width = image_data["height"], image_data["width"]
            array = np.frombuffer(image_data["data"], dtype=np.uint8).reshape((height, width, 3))

            # BMP stores images in BGR order and bottom-up
            array = array[:, :, ::-1]
            array = np.flipud(array)
        else:
            image = Image.open(path)
            original_width, original_height = image.size

            max_width = 484
            max_height = 300

            ratio = min(max_width / original_width, max_height / original_height)

            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            array = np.array(image)

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')

        self.image = array
        image1 = Image.fromarray(self.image)

        image1 = ImageTk.PhotoImage(image1)

        self.panelA = Label(image=image1, borderwidth=5)
        self.panelA.image = image1
        self.panelA.grid(
            row= 1,
            column=1,
            rowspan= 13,
            columnspan= 3,
            padx=2,
            pady=10
        )

        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        return self.image

    def display_org_image_PanelA(self):
        image1 = Image.fromarray(self.image)
        image1 = ImageTk.PhotoImage(image1)
        self.panelA = Label(image=image1, borderwidth=5)
        self.panelA.image = image1
        self.panelA.grid(
            row= 1,
            column=1,
            rowspan= 13,
            columnspan= 3,
            padx=2,
            pady=10
        )

    def convert_to_grayscale(self, isPanelA = False):
        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')

        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        if self.panelA and not isPanelA:
            self.display_org_image_PanelA()

            # Convert the image to a NumPy array for manipulation
            np_image = np.array(self.image)
            
            # Apply the luminosity formula
            gray_values = get_grayscale_values(np_image)

            # Convert back to an integer type
            gray_image = np.uint8(gray_values)

            gray_image= Image.fromarray(gray_image)
            
            gray_image = ImageTk.PhotoImage(gray_image)
            
            self.panelB = Label(image=gray_image, borderwidth=5)
            self.panelB.image = gray_image
            self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
            
            return gray_image
        elif isPanelA:
            np_image = np.array(self.image)
            
            # Apply the luminosity formula
            gray_values = get_grayscale_values(np_image)

            # Convert back to an integer type
            gray_image = np.uint8(gray_values)
            gray_image= Image.fromarray(gray_image)
            
            gray_image = ImageTk.PhotoImage(gray_image)
            
            self.panelA = Label(image=gray_image, borderwidth=5)
            self.panelA.image = gray_image
            self.panelA.grid(
                row= 1,
                column=1,
                rowspan= 13,
                columnspan= 3,
                padx=2,
                pady=10
            )
            
            return gray_image
        else:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")

    def apply_ordered_dithering(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        gray_image_conv = self.convert_to_grayscale(True)

        # Define a 4x4 Bayer matrix
        bayer_matrix = np.array([[0, 8, 2, 10],
                                [12, 4, 14, 6],
                                [3, 11, 1, 9],
                                [15, 7, 13, 5]]) / 15.0 * 255.0

        np_image = np.array(self.image)

        gray_values = get_grayscale_values(np_image)

        # Convert back to an integer type
        gray_image = np.uint8(gray_values)

        gray_image= Image.fromarray(gray_image)

        # rows, cols = gray_image.size
        rows, cols = (np.array(gray_values)).shape
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

        dithered_image = ImageTk.PhotoImage(dithered_image)
        self.panelB = Label(image=dithered_image, borderwidth=5)
        self.panelB.image = dithered_image
        self.panelB.grid(row= 1, column=4 , rowspan= 13, pady=10)

        return gray_image_conv, dithered_image

    def apply_auto_level(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()
        image1 = Image.fromarray(self.image)
        
        # Convert the image to a NumPy array for easier manipulation
        img_array = np.array(image1)

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

        auto_leveled_image = ImageTk.PhotoImage(auto_leveled_image)
        self.panelB = Label(image=auto_leveled_image, borderwidth=5)
        self.panelB.image = auto_leveled_image
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
        return auto_leveled_image

    def show_huffman_stats(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')

        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        # Convert the image to a NumPy array for manipulation
        np_image = np.array(self.image)
        
        # Apply the luminosity formula
        gray_values = get_grayscale_values(np_image)
        
        # Convert back to an integer type
        gray_image = np.uint8(gray_values)

        img_array = np.array(gray_image)
        histogram = Counter(img_array.flatten())
        total_pixels = sum(histogram.values())

        probabilities = {k: v / total_pixels for k, v in histogram.items()}
        img_entropy = -sum(p * np.log2(p) for p in probabilities.values())

        huffman_tree = build_huffman_tree(histogram)
        avg_code_length = calculate_average_code_length(huffman_tree, histogram)

        messagebox.showinfo("Huffman Stats", f"Entropy: {img_entropy:.2f} bits\nAverage Huffman Code Length: {avg_code_length:.2f} bits")

    def negative(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')

        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        neg1= Image.fromarray(255 - self.image)
        
        neg1= ImageTk.PhotoImage(neg1)
        
        self.panelB = Label(image=neg1, borderwidth=5)
        self.panelB.image = neg1
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
        
        return 255 - self.image

    def extract_red(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        red = get_color_component(self.image, 0)
        red1 = Image.fromarray(red)
        
        red1= ImageTk.PhotoImage(red1)
        
        self.panelB = Label(image=red1, borderwidth=5)
        self.panelB.image = red1
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
            
        return red

    def extract_green(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return
        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        green = get_color_component(self.image, 1)
        
        green1 = Image.fromarray(green)
        
        green1= ImageTk.PhotoImage(green1)
        
        self.panelB = Label(image=green1, borderwidth=5)
        self.panelB.image = green1
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
        
        return green

    def extract_blue(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        blue = get_color_component(self.image, 2)
        
        blue1 = Image.fromarray(blue)
        
        blue1= ImageTk.PhotoImage(blue1)
        
        self.panelB = Label(image=blue1, borderwidth=5)
        self.panelB.image = blue1
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
            
        return blue

    def detect_edges(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        # Convert the image to a NumPy array for manipulation
        np_image = np.array(self.image)
        
        # Apply the luminosity formula
        gray_values = get_grayscale_values(np_image)
        
        # Convert back to an integer type
        gray_image = np.uint8(gray_values)

        # Apply manual thresholding: Set pixels to max_value if above the threshold, otherwise to 0
        binary_image = np.where(gray_image > 127, 255, 0).astype(np.uint8)

        edged = cv2.Canny(binary_image, 50, 100)
        edged1 = Image.fromarray(edged)
        
        edged1= ImageTk.PhotoImage(edged1)
        
        self.panelB = Label(image=edged1, borderwidth=5)
        self.panelB.image = edged1
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
        
        return edged

    def pencil_sketch_effect(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        # Convert the image to a NumPy array for manipulation
        np_image = np.array(self.image)

        # Apply the luminosity formula
        gray_values = get_grayscale_values(np_image)
        
        # Convert back to an integer type
        gray_image = np.uint8(gray_values)

        img_invert = invert_image(gray_image)
        img_smoothing = gaussian_blur(img_invert, 21, sigma=0)
        new_pencil_img = divide_images(gray_image, 255 - img_smoothing, scale=255)
        new_pencil_image= Image.fromarray(new_pencil_img)
        
        new_pencil_image= ImageTk.PhotoImage(new_pencil_image)
        
        self.panelB = Label(image=new_pencil_image, borderwidth=5)
        self.panelB.image = new_pencil_image
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)

        return new_pencil_img

    def color_pencil_sketch_effect(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        img_invert = invert_image(self.image)
        img_smoothing = gaussian_blur(img_invert, 21, sigma=0)
        new_color_pencil_img = divide_images(self.image, 255 - img_smoothing, scale=255)
        new_color_pencil_image= Image.fromarray(new_color_pencil_img)
        
        new_color_pencil_image= ImageTk.PhotoImage(new_color_pencil_image)
        
        self.panelB = Label(image=new_color_pencil_image, borderwidth=5)
        self.panelB.image = new_color_pencil_image
        self.panelB.grid(row= 1, column=4 , rowspan= 13,columnspan= 3, padx=2, pady=10)
        
        return new_color_pencil_img

    def apply_sepia_filter(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')
        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        # Convert the image to a NumPy array for manipulation
        np_image = np.array(self.image)

        # Apply sepia tone
        sepia_image = get_sepia_values(np_image)

        sepia_image = np.clip(sepia_image, 0, 255)

        sepia_image = Image.fromarray(np.uint8(sepia_image))
        sepia_image = ImageTk.PhotoImage(sepia_image)

        self.panelB = Label(image=sepia_image, borderwidth=5)
        self.panelB.image = sepia_image
        self.panelB.grid(row=1, column=4, rowspan=13, columnspan=3, padx=2, pady=10)

        return sepia_image

    def apply_gaussian_noise(self):
        if self.image is None:
            messagebox.showinfo("Information", "No image loaded. Please open an image first.")
            return

        if hasattr(self, 'panelA') and self.panelA is not None:
            self.panelA.config(image='')

        if hasattr(self, 'panelB') and self.panelB is not None:
            self.panelB.config(image='')

        self.display_org_image_PanelA()

        np_image = np.array(self.image)

        # Apply Gaussian noise
        noisy_image_array = add_gaussian_noise_to_image(np_image)

        noisy_image = Image.fromarray(np.uint8(noisy_image_array))
        noisy_image = ImageTk.PhotoImage(noisy_image)

        self.panelB = Label(image=noisy_image, borderwidth=5)
        self.panelB.image = noisy_image
        self.panelB.grid(row=1, column=6, rowspan=13, columnspan=2, padx=(0,20), pady = 20)

        return noisy_image_array

    def exit_app(self):
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = MiniPhotoshop(root)
    root.mainloop()
