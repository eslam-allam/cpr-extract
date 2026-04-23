import cv2
import numpy as np
from rq import get_current_job
from core.extract import extract_data
import worker_init



def process_cpr_task(front: bytes, back: bytes):
    # Initialize OCR with the new mkldnn parameter
    #
    ocr = worker_init.get_ocr()

    final = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
        "nationality": None,
    }

    for image_bytes in [front, back]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del nparr

        if img is None:
            job = get_current_job()
            if job:
                job.meta['error'] = 'Invalid image'
                job.save_meta()
            raise

        res = extract_data(ocr.predict(img), img.shape[0])
        del img

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
        if final["nationality"] is None and res["nationality"] is not None:
            final["nationality"] = res["nationality"]

    return final
