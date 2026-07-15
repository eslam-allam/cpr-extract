import os
import pickle
import requests
from rq import get_current_job
from core.extract import extract_data

OCR_SERVICE_URL = os.getenv("OCR_SERVICE_URL", "http://ocr-service:5000/predict")


def process_cpr_task(front: bytes, back: bytes):
    final = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
        "nationality": None,
    }

    job = get_current_job()

    for side, image_bytes in [("front", front), ("back", back)]:
        if not image_bytes or len(image_bytes) < 100:
            if job:
                job.meta["error"] = f"Invalid or empty {side} image"
                job.save_meta()
            raise ValueError(f"The provided {side} image payload is invalid or empty.")

        try:
            files = {"image": (f"{side}.jpg", image_bytes, "image/jpeg")}
            response = requests.post(OCR_SERVICE_URL, files=files, timeout=30)
            response.raise_for_status()

            payload = pickle.loads(response.content)
            ocr_results = payload["results"]
            image_height = payload["shape_0"]

        except Exception as e:
            error_msg = f"OCR Service communication failure on {side} image: {str(e)}"
            if job:
                job.meta["error"] = error_msg
                job.save_meta()
            raise RuntimeError(error_msg) from e

        try:
            res = extract_data(ocr_results, image_height)
        except Exception as e:
            error_msg = f"Parsing extraction failed on {side} image: {str(e)}"
            if job:
                job.meta["error"] = error_msg
                job.save_meta()
            raise RuntimeError(error_msg) from e

        if res.get("cpr_verified"):
            final["cpr"] = res["cpr"]
            final["cpr_verified"] = True
        elif not final["cpr"]:
            final["cpr"] = res.get("cpr")

        if res.get("arabic_name"):
            final["arabic_name"] = res["arabic_name"]
        if res.get("english_name"):
            final["english_name"] = res["english_name"]
        if res.get("dob"):
            final["dob"] = res["dob"]
        if final["nationality"] is None and res.get("nationality") is not None:
            final["nationality"] = res["nationality"]

    return final
