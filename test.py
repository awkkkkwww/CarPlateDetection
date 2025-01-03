from paddleocr import PaddleOCR
import re
import cv2
import numpy as np

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
    识别车牌并显示锁定后的车牌照片
    :param image_path: 输入图片路径
    :return: 返回识别到的车牌信息列表
    """
    # 定位车牌
    results = locate_license_plate(image_path)

    # 读取原始图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图片，请检查路径是否正确。")

    # 显示每个车牌区域
    for i, result in enumerate(results):
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

        # 显示车牌区域
        cv2.imshow(f"License Plate {i + 1} ({plate_color})", plate_image)
        cv2.waitKey(0)  # 等待用户按下任意键关闭窗口

        # 输出车牌信息
        print(f"车牌 {i + 1}:")
        print(f"  颜色: {plate_color}")
        print(f"  文本: {result['text']}")
        print(f"  置信度: {result['confidence']:.2f}")
        print(f"  位置坐标: {result['coords']}")

    cv2.destroyAllWindows()  # 关闭所有 OpenCV 窗口

    return results

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

# 测试代码
if __name__ == "__main__":
    image_path = r"D:\python_leanling_code\TXCL_end\ph\11.jpg"  # 替换为你的图片路径

    # 识别车牌并显示锁定后的车牌照片
    results = recognize_license_plate(image_path)