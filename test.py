from paddleocr import PaddleOCR
import re

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

def locate_license_plate(image_path):
    """
    定位图片中的车牌位置
    :param image_path: 图片路径
    :return: 返回符合车牌格式的文本
    """
    # 使用 PaddleOCR 进行识别
    result = ocr.ocr(image_path, det=True, rec=True)

    # 提取符合车牌格式的文本
    valid_candidates = []
    for line in result[0]:
        text = line[1][0]
        if is_valid_license_plate(text):
            valid_candidates.append(text)
    return valid_candidates

# 测试代码
if __name__ == "__main__":
    image_path = r"E:\photo\car\2.jpg"  # 替换为你的图片路径
    results = locate_license_plate(image_path)
    for text in results:
        print(f"有效车牌: {text}")