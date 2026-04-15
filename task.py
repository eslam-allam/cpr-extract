import cv2
import os
import numpy as np
from core.extract import extract_data

def process_cpr_task(
    front: bytes, back: bytes, mkldnn=False, model_source_check=False
):
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = str(
        not model_source_check
    )

    from paddleocr import PaddleOCR

    # Apply environment silencers even if verbose is off

    # Initialize OCR with the new mkldnn parameter
    ocr = PaddleOCR(enable_mkldnn=mkldnn, lang="ar", ocr_version="PP-OCRv5")

    final = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
    }

    for bytes in [front, back]:
        nparr = np.frombuffer(bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        res = extract_data(ocr.predict(img), img.shape[0])

        if res["cpr_verified"]:
            final["cpr"], final["cpr_verified"] = res["cpr"], True
        elif not final["cpr"]:
            final["cpr"] = res["cpr"]

        if res["arabic_name"]:
            final["arabic_name"] = res["arabic_name"]
        if res["english_name"]:
            final["english_name"] = res["english_name"]
        if res["dob"]:
            final["dob"] = res["dob"]

    return final
