from paddleocr import PaddleOCR
import os

ocr = None

def initialize():
    global ocr
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    ocr = PaddleOCR(enable_mkldnn=False, lang="ar", ocr_version="PP-OCRv5")

def get_ocr():
    return ocr
