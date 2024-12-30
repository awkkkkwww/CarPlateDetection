from paddleocr import PaddleOCR

# 初始化 OCR
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

def locate_license_plate(image_path):
    """
    定位图片中的车牌位置
    :param image_path: 图片路径
    :return: 返回 OCR 识别结果
    """
    # 使用 PaddleOCR 进行识别
    result = ocr.ocr(image_path, det=True, rec=True)
    return result

# 测试代码
if __name__ == "__main__":
    image_path = r"E:\photo\car\2.jpg"  # 替换为你的图片路径
    results = locate_license_plate(image_path)
    for line in results[0]:
        print(line)