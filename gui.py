import tkinter as tk
from tkinter import filedialog, Scale, messagebox
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_processing import colorgrad


class EdgeDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Ứng dụng phát hiện biên ảnh màu")

        self.image = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        self.select_button = tk.Button(self.master, text="Chọn ảnh", command=self.select_image)
        self.select_button.pack(pady=10)

        self.threshold_scale = Scale(self.master, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Ngưỡng",
                                     command=self.update_image)
        self.threshold_scale.set(0.5)
        self.threshold_scale.pack(pady=10)

        self.mask_var = tk.StringVar(value="sobel")
        self.sobel_radio = tk.Radiobutton(self.master, text="Sobel", variable=self.mask_var, value="sobel",
                                          command=self.update_image)
        self.prewitt_radio = tk.Radiobutton(self.master, text="Prewitt", variable=self.mask_var, value="prewitt",
                                            command=self.update_image)
        self.sobel_radio.pack()
        self.prewitt_radio.pack()

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            try:
                self.image = cv2.imread(file_path)
                if self.image is None:
                    raise ValueError("Không thể đọc file ảnh")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.update_image()
            except Exception as e:
                messagebox.showerror("Lỗi",
                                     f"Không thể mở file ảnh: {str(e)}\nVui lòng kiểm tra lại đường dẫn và định dạng file.")
                self.image = None

    def update_image(self, *args):
        if self.image is not None:
            threshold = self.threshold_scale.get()
            mask_type = self.mask_var.get()

            VG, _, _ = colorgrad(self.image, mask_type, threshold)

            self.axs[0].clear()
            self.axs[0].imshow(self.image)
            self.axs[0].set_title("Ảnh gốc")
            self.axs[0].axis('off')

            self.axs[1].clear()
            self.axs[1].imshow(VG, cmap='gray')
            self.axs[1].set_title(f"Phát hiện biên ({mask_type})")
            self.axs[1].axis('off')

            self.canvas.draw()
        else:
            self.axs[0].clear()
            self.axs[1].clear()
            self.axs[0].text(0.5, 0.5, "Chưa chọn ảnh", ha='center', va='center')
            self.axs[1].text(0.5, 0.5, "Chưa chọn ảnh", ha='center', va='center')
            self.canvas.draw()