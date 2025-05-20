import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
from matplotlib.pyplot import margins
from scipy.ndimage import convolve
import cv2

# Kernel definitions for existing filters
kernels = {
    "Horizontal Sobel": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "Vertical Sobel": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    "Left Diagonal": np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]),
    "Right Diagonal": np.array([[-1, -1, 1], [-1, 1, -1], [1, -1, -1]]),
    "Edge Detection": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Box Blur": (1 / 9.0) * np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]),
    "Gaussian Blur (SciPy)": (1 / 16.0) * np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
}

# Noise functions
def add_salt_and_pepper_noise(image, noise_ratio=0.5):
    noisy_image = image.copy()
    h, w, c = noisy_image.shape
    num_noisy_pixels = int(h * w * noise_ratio)
    for _ in range(num_noisy_pixels):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        if np.random.rand() < 0.5:
            noisy_image[y, x] = [0, 0, 0]
        else:
            noisy_image[y, x] = [255, 255, 255]
    return noisy_image

def add_random_noise(image, intensity=25):
    noisy_image = image.copy()
    noise = np.random.randint(-intensity, intensity + 1, noisy_image.shape)
    noisy_image = np.clip(noisy_image + noise, 0, 255).astype(np.uint8)
    return noisy_image

class ImageFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Filter Studio")
        self.root.geometry("800x600")
        self.root.configure(bg="#06275c")  # Soft off-white background

        # Configure ttk style for buttons and dropdown
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "TButton",
            font=("Roboto", 12, "bold"),
            padding=12,
            background="#eed055",
            foreground="#06275c",
            borderwidth=0,
            bordercolor="#06275c",
            relief="solid",
            anchor="center",
            justify="center",
            focuscolor="#eed055"
        )
        style.map(
            "TButton",
            background=[("active", "#FFF176"), ("disabled", "#eed055")],
            foreground=[("active", "#06275c"), ("disabled", "#06275c")]
        )
        style.configure(
            "TLabel",
            background="#06275c",  # Light mint green for labels
            foreground="#eed055",  # Deep navy blue text
            font=("Comic Sans MS", 16)
        )
        style.configure(
            "TCombobox",
            padding=10,
            background="#eed055",
            foreground="#06275c",
            fieldbackground="#b2cdf7",
            borderwidth=0,
            relief="solid"
        )
        style.map(
            "TCombobox",
            background=[("disabled", "#eed055")],
            foreground=[("disabled", "#06275c")],
            fieldbackground=[("disabled", "#b2cdf7")]
        )


        # GUI Elements
        self.label = tk.Label(
            root,
            text="Image Filter Studio",
            font=("Comic Sans MS", 20,"bold"),
            bg="#06275c",
            fg="#eed055"
        )
        self.label.pack(pady=25)

        # Image display frames
        self.frame_images = tk.Frame(root, bg="#06275c")
        self.frame_images.pack(pady=25)

        self.label_original = ttk.Label(self.frame_images, text="Original Image")
        self.label_original.grid(row=0, column=0, padx=20, pady=5)
        self.label_filtered = ttk.Label(self.frame_images, text="Filtered Image")
        self.label_filtered.grid(row=0, column=1, padx=20, pady=5)

        # Canvas with yellow borders
        self.canvas_original = tk.Label(
            self.frame_images,
            bg="#06275c",  # Light mint green background
            highlightbackground="#06275c",  # Yellow border
        )
        self.canvas_original.grid(row=1, column=0, padx=20, pady=15)
        self.canvas_filtered = tk.Label(
            self.frame_images,
            bg="#06275c",
            highlightbackground="#06275c",
        )
        self.canvas_filtered.grid(row=1, column=1, padx=20, pady=15)

        # Buttons and Filter Selection
        self.button_frame = tk.Frame(root, bg="#06275c")
        self.button_frame.pack(pady=25)

        self.btn_load = ttk.Button(self.button_frame, text="Load Image", command=self.load_image,)

        self.btn_load.grid(row=0, column=0, padx=15)

        self.filter_var = tk.StringVar(value="None")
        self.filters = [
            "None", "Horizontal Sobel", "Vertical Sobel", "Left Diagonal", "Right Diagonal",
            "Edge Detection", "Sharpen", "Box Blur", "Gaussian Blur (SciPy)", "Bilateral Filter",
            "Median Blur", "Gaussian Blur (OpenCV)", "Blur (OpenCV)", "Box Filter (OpenCV)", "Laplacian",
            "Gaussian Noise", "Salt and Pepper Noise", "Random Noise"
        ]
        self.filter_menu = ttk.Combobox(
            self.button_frame, textvariable=self.filter_var, values=self.filters, state="readonly",
        )
        self.filter_menu.grid(row=0, column=1, padx=15)

        self.btn_apply = ttk.Button(self.button_frame, text="Apply Filter", command=self.apply_filter)
        self.btn_apply.grid(row=0, column=2, padx=15)

        self.btn_save = ttk.Button(self.button_frame, text="Save Filtered Image", command=self.save_image)
        self.btn_save.grid(row=0, column=3, padx=15)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if self.image_path:
            try:
                self.original_image = Image.open(self.image_path)
                display_image = self.original_image.copy()
                display_image.thumbnail((300, 300))
                self.photo_original = ImageTk.PhotoImage(display_image)
                self.canvas_original.configure(image=self.photo_original)
                self.canvas_filtered.configure(image='')
                self.filtered_image = None
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def apply_filter(self):
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first!")
            return

        filter_type = self.filter_var.get()
        self.filtered_image = self.original_image.copy()

        try:
            if filter_type in kernels:
                img_array = np.array(self.filtered_image.convert("L")).astype(float)
                filtered_array = convolve(img_array, kernels[filter_type])
                filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)
                self.filtered_image = Image.fromarray(filtered_array)
            elif filter_type in [
                "Bilateral Filter", "Median Blur", "Gaussian Blur (OpenCV)",
                "Blur (OpenCV)", "Box Filter (OpenCV)", "Laplacian",
                "Gaussian Noise", "Salt and Pepper Noise", "Random Noise"
            ]:
                img_array = np.array(self.filtered_image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                if filter_type == "Bilateral Filter":
                    filtered_array = cv2.bilateralFilter(img_array, 20, 75, 75)
                elif filter_type == "Median Blur":
                    filtered_array = cv2.medianBlur(img_array, ksize=9)
                elif filter_type == "Gaussian Blur (OpenCV)":
                    filtered_array = cv2.GaussianBlur(img_array, (5, 5), sigmaX=10)
                elif filter_type == "Blur (OpenCV)":
                    filtered_array = cv2.blur(img_array, (5, 5))
                elif filter_type == "Box Filter (OpenCV)":
                    filtered_array = cv2.boxFilter(img_array, ddepth=-1, ksize=(5, 5), normalize=True)
                elif filter_type == "Laplacian":
                    filtered_array = cv2.Laplacian(img_array, ddepth=cv2.CV_64F)
                    filtered_array = cv2.convertScaleAbs(filtered_array)
                elif filter_type == "Gaussian Noise":
                    mean = 5
                    std = 25
                    gaussian = np.random.normal(mean, std, img_array.shape).astype(np.uint8)
                    filtered_array = img_array + gaussian
                    filtered_array = np.clip(filtered_array, 0, 255).astype(np.uint8)
                elif filter_type == "Salt and Pepper Noise":
                    filtered_array = add_salt_and_pepper_noise(img_array, noise_ratio=0.5)
                elif filter_type == "Random Noise":
                    filtered_array = add_random_noise(img_array, intensity=25)
                filtered_array = cv2.cvtColor(filtered_array, cv2.COLOR_BGR2RGB)
                self.filtered_image = Image.fromarray(filtered_array)
            elif filter_type == "None":
                self.filtered_image = self.original_image.copy()

            display_filtered = self.filtered_image.copy()
            display_filtered.thumbnail((300, 300))
            self.photo_filtered = ImageTk.PhotoImage(display_filtered)
            self.canvas_filtered.configure(image=self.photo_filtered)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {e}")

    def save_image(self):
        if not self.filtered_image:
            messagebox.showwarning("Warning", "No filtered image to save!")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )
        if save_path:
            try:
                self.filtered_image.save(save_path)
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageFilterApp(root)
    root.mainloop()