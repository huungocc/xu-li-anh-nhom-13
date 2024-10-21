import tkinter as tk
from tkinter import filedialog, Scale, messagebox, colorchooser
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from image_processing import colorgrad, create_mask, apply_background


class EdgeDetectionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Ứng dụng phát hiện biên ảnh màu và tách nền ảnh màu")

        self.image = None
        self.processed_image = None
        self.mask = None
        self.background_image = None

        self.create_widgets()

    def create_widgets(self):
        self.select_button = tk.Button(self.master, text="Chọn ảnh", command=self.select_image)
        self.select_button.pack(pady=10)

        self.threshold_scale = Scale(self.master, length=400, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, label="Ngưỡng",
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

        # Tách nền
        self.segment_button = tk.Button(self.master, text="Tách đối tượng", command=self.segment_object)
        self.segment_button.pack(pady=10)

        self.background_frame = tk.Frame(self.master)
        self.background_frame.pack(pady=10)

        self.color_button = tk.Button(self.background_frame, text="Chọn màu nền", command=self.choose_background_color)
        self.color_button.pack(side=tk.LEFT, padx=5)

        self.image_button = tk.Button(self.background_frame, text="Chọn ảnh nền", command=self.choose_background_image)
        self.image_button.pack(side=tk.LEFT, padx=5)

        self.transparent_button = tk.Button(self.background_frame, text="Nền trong suốt",
                                            command=self.set_transparent_background)
        self.transparent_button.pack(side=tk.LEFT, padx=5)

        self.export_button = tk.Button(self.master, text="Xuất ảnh", command=self.export_image)
        self.export_button.pack(pady=10)

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
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
            self.mask = create_mask(VG, threshold)

            self.axs[0].clear()
            self.axs[0].imshow(self.image)
            self.axs[0].set_title("Ảnh gốc")
            self.axs[0].axis('off')

            self.axs[1].clear()
            self.axs[1].imshow(VG, cmap='gray')
            self.axs[1].set_title(f"Phát hiện biên ({mask_type})")
            self.axs[1].axis('off')

            self.axs[2].clear()
            self.axs[2].imshow(self.mask, cmap='gray')
            self.axs[2].set_title("Mặt nạ")
            self.axs[2].axis('off')

            self.canvas.draw()
        else:
            self.axs[0].clear()
            self.axs[1].clear()
            self.axs[0].text(0.5, 0.5, "Chưa chọn ảnh", ha='center', va='center')
            self.axs[1].text(0.5, 0.5, "Chưa chọn ảnh", ha='center', va='center')
            self.canvas.draw()

    def segment_object(self):
        if self.image is not None and self.mask is not None:
            self.processed_image = apply_background(self.image, self.mask)
            self.update_display()

    def choose_background_color(self):
        color = colorchooser.askcolor(title="Chọn màu nền")[0]
        if color:
            self.background_image = None
            self.processed_image = apply_background(self.image, self.mask, background_color=color)
            self.update_display()

    def choose_background_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")])
        if file_path:
            self.background_image = cv2.imread(file_path)
            self.background_image = cv2.cvtColor(self.background_image, cv2.COLOR_BGR2RGB)
            self.processed_image = apply_background(self.image, self.mask, background_image=self.background_image)
            self.update_display()

    def set_transparent_background(self):
        self.background_image = None
        self.processed_image = apply_background(self.image, self.mask)
        self.update_display()

    def export_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"),
                                                                ("JPEG files", "*.jpg"),
                                                                ("All files", "*.*")])
            if file_path:
                try:
                    if self.processed_image.shape[2] == 4:  # Image with alpha channel (RGBA)
                        cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGBA2BGRA))
                    else:  # RGB image
                        cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                    messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {str(e)}")
        else:
            messagebox.showwarning("Cảnh báo", "Không có ảnh để xuất. Vui lòng xử lý ảnh trước.")

    def update_display(self):
        if self.processed_image is not None:
            self.axs[2].clear()
            self.axs[2].imshow(self.processed_image)
            self.axs[2].set_title("Kết quả tách nền")
            self.axs[2].axis('off')
            self.canvas.draw()

            # Enable the export button when there's a processed image
            self.export_button.config(state=tk.NORMAL)
        else:
            # Disable the export button when there's no processed image
            self.export_button.config(state=tk.DISABLED)