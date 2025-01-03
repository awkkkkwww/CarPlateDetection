import tkinter as tk
from tkinter import filedialog, messagebox
from paddleocr import PaddleOCR
import re
import cv2
import numpy as np
from PIL import Image, ImageTk

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

# 车牌正则表达式
plate_patterns = [
    r"^[\u4e00-\u9fff][A-Z]·[A-Z0-9]{5}$",  # 中国车牌
    r"^[A-Z]{1,3}[0-9]{1,4}[A-Z0-9]{0,3}$",  # 国际车牌
    r"^粤Z·[A-Z0-9]{4}[港澳]$",  # 粤Z车牌
    r"^[\u4e00-\u9fff][A-Z]·[DF][A-Z0-9]{5}$"  # 新能源车牌
]
plate_pattern = re.compile("|".join(plate_patterns))


def is_valid_license_plate(text):
    """检查文本是否符合车牌格式"""
    return plate_pattern.match(text) is not None


def detect_plate_color(plate_image):
    """
    识别车牌颜色
    :param plate_image: 车牌图片
    :return: 返回车牌颜色（蓝牌、绿牌、黄牌、白牌、黑牌）
    """
    # 将车牌图片转换为 HSV 颜色空间
    hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    blue_lower = np.array([100, 50, 50])  # 蓝色范围
    blue_upper = np.array([130, 255, 255])

    green_lower = np.array([35, 50, 50])  # 绿色范围
    green_upper = np.array([85, 255, 255])

    yellow_lower = np.array([20, 50, 50])  # 黄色范围
    yellow_upper = np.array([35, 255, 255])

    white_lower = np.array([0, 0, 200])  # 白色范围
    white_upper = np.array([180, 30, 255])

    black_lower = np.array([0, 0, 0])  # 黑色范围
    black_upper = np.array([180, 255, 100])

    # 创建颜色掩码
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    white_mask = cv2.inRange(hsv_image, white_lower, white_upper)
    black_mask = cv2.inRange(hsv_image, black_lower, black_upper)

    # 计算各颜色的像素数量
    blue_pixels = cv2.countNonZero(blue_mask)
    green_pixels = cv2.countNonZero(green_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    white_pixels = cv2.countNonZero(white_mask)
    black_pixels = cv2.countNonZero(black_mask)

    # 判断车牌颜色
    max_pixels = max(blue_pixels, green_pixels, yellow_pixels, white_pixels, black_pixels)
    if max_pixels == blue_pixels:
        return "蓝牌"
    elif max_pixels == green_pixels:
        return "绿牌"
    elif max_pixels == yellow_pixels:
        return "黄牌"
    elif max_pixels == white_pixels:
        return "白牌"
    elif max_pixels == black_pixels:
        return "黑牌"
    else:
        return "未知"


def locate_license_plate(image_path):
    """
    定位图片中的车牌位置
    :param image_path: 图片路径
    :return: 返回一个列表，每个元素是一个字典，包含车牌文本、置信度和位置坐标
    """
    # 使用 PaddleOCR 进行识别
    result = ocr.ocr(image_path, det=True, rec=True)

    # 提取有效车牌信息
    valid_candidates = []
    for line in result[0]:
        coords, (text, confidence) = line[0], line[1]
        if is_valid_license_plate(text) and confidence > 0.9:
            valid_candidates.append({
                "text": text,
                "confidence": confidence,
                "coords": [tuple(map(int, point)) for point in coords]
            })

    # 按置信度排序
    valid_candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return valid_candidates


def recognize_license_plate(image_path):
    """
    识别车牌并返回结果
    :param image_path: 输入图片路径
    :return: 返回识别到的车牌信息列表
    """
    # 定位车牌
    results = locate_license_plate(image_path)

    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图片，请检查路径是否正确。")

    # 处理每个车牌区域
    plate_results = []
    for result in results:
        coords = result["coords"]
        # 获取车牌区域的边界框
        x_min = min(point[0] for point in coords)
        x_max = max(point[0] for point in coords)
        y_min = min(point[1] for point in coords)
        y_max = max(point[1] for point in coords)

        # 截取车牌区域
        plate_image = image[y_min:y_max, x_min:x_max]

        # 识别车牌颜色
        plate_color = detect_plate_color(plate_image)

        # 将车牌图片转换为 PIL 格式
        plate_image_pil = Image.fromarray(cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB))

        # 存储结果
        plate_results.append({
            "image": plate_image_pil,
            "color": plate_color,
            "text": result["text"],
            "confidence": result["confidence"],
            "coords": result["coords"]
        })

    return plate_results, image

class LicensePlateApp:
    def __init__(self, root):#GUI显示区域
        self.root = root
        self.root.title("车牌识别系统")

        # 创建 GUI 组件
        self.label = tk.Label(root, text="车牌识别系统", font=("Arial", 16))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="选择图片", command=self.select_image)
        self.select_button.pack(pady=10)

        # 原图片显示区域
        self.original_image_label = tk.Label(root)
        self.original_image_label.pack(pady=10)

        # 结果显示区域
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=10)

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车牌识别系统")

        # 创建 GUI 组件
        self.label = tk.Label(root, text="车牌识别系统", font=("Arial", 16))
        self.label.pack(pady=10)

        self.select_button = tk.Button(root, text="选择图片", command=self.select_image)
        self.select_button.pack(pady=10)

        # 原图片显示区域
        self.original_image_label = tk.Label(root)
        self.original_image_label.pack(pady=10)

        # 结果显示区域
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(pady=10)

    def select_image(self):
        """选择图片并显示识别结果"""
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if not file_path:
            return

        # 清空之前的结果
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        # 识别车牌
        try:
            plate_results, original_image = recognize_license_plate(file_path)
            if not plate_results:
                messagebox.showinfo("提示", "未检测到车牌")
                return

            # 显示原图片
            original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            original_image_pil = original_image_pil.resize((400, 300), Image.ANTIALIAS)
            original_image_tk = ImageTk.PhotoImage(original_image_pil)
            self.original_image_label.config(image=original_image_tk)
            self.original_image_label.image = original_image_tk  # 保持引用，避免被垃圾回收

            # 显示每个车牌的结果
            for i, result in enumerate(plate_results):
                # 显示车牌图片
                plate_image = result["image"].resize((200, 100), Image.ANTIALIAS)
                plate_image_tk = ImageTk.PhotoImage(plate_image)
                image_label = tk.Label(self.result_frame, image=plate_image_tk)
                image_label.image = plate_image_tk  # 保持引用，避免被垃圾回收
                image_label.grid(row=i, column=0, padx=10, pady=10)

                # 显示车牌信息
                info_text = (f"车牌 {i + 1}:\n"
                             f"  车牌颜色: {result['color']}\n"
                             f"  车牌号码: {result['text']}\n"
                             f"  置信度: {result['confidence']:.2f}\n"
                             f"  位置坐标: {result['coords']}")
                info_label = tk.Label(self.result_frame, text=info_text, justify=tk.LEFT)
                info_label.grid(row=i, column=1, padx=10, pady=10)

        except Exception as e:
            messagebox.showerror("错误", f"发生错误: {e}")


# 运行 GUI 程序
if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()