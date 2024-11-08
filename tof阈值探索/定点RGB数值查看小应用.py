import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")

        # 初始化变量
        self.image_path = ""
        self.image = None

        # 创建GUI元素
        self.create_widgets()

    def create_widgets(self):
        # 选择图片按钮
        select_button = tk.Button(self.root, text="选择图片", command=self.load_image)
        select_button.pack(pady=10)

        # 显示图像信息的标签
        self.info_label = tk.Label(self.root, text="")
        self.info_label.pack()

        # 显示图像的画布
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(expand=True, fill=tk.BOTH)  # 使画布充满整个窗口

        # 点击画布事件
        self.canvas.bind("<Button-1>", self.show_pixel_info)

    def load_image(self):
        # 打开文件对话框
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

        if file_path:
            # 加载图像
            self.image_path = file_path
            self.image = Image.open(self.image_path)

            # 显示图像
            self.display_image()

            # 更新图像信息
            self.update_info_label()

    def display_image(self):
        # 等比例缩小图像
        width, height = self.image.size
        new_width = min(width, 800)
        new_height = int(height * (new_width / width))
        resized_image = self.image.resize((new_width, new_height), Image.LANCZOS)



        # 将图像转换为Tkinter PhotoImage对象
        tk_image = ImageTk.PhotoImage(resized_image)

        # 在画布上显示图像
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

        # 保留对PhotoImage对象的引用，以避免垃圾回收
        self.canvas.image = tk_image

    def update_info_label(self):
        # 显示图像信息
        info_text = f"图像大小: {self.image.width} x {self.image.height}"
        self.info_label.config(text=info_text)

    def show_pixel_info(self, event):
        # 获取点击的像素坐标
        x, y = event.x, event.y

        # 获取像素RGB值
        rgb = self.image.getpixel((x, y))

        # 在标签中显示像素信息
        pixel_info_text = f"坐标: ({x}, {y})   RGB: {rgb}"
        self.info_label.config(text=pixel_info_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.geometry("1920x1080")  # 设置窗口的初始大小为800x600
    root.mainloop()
