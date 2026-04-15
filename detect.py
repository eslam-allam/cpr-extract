import sys
import re
import json
import cv2
from paddleocr import PaddleOCR


def validate_bahrain_cpr(cpr_str):
    if not cpr_str:
        return False
    clean = re.sub(r"\D", "", cpr_str)
    if len(clean) != 9:
        return False
    digits = [int(d) for d in clean]
    weights = [7, 6, 5, 4, 3, 2, 7, 6]
    total = sum(digits[i] * weights[i] for i in range(8))
    check = (11 - (total % 11)) % 11
    if check == 10:
        check = 0
    return check == digits[8]


def extract_data(results, w, h):
    fields = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
    }
    blocks = []

    for page in list(results):
        boxes = page.get("dt_polys", [])
        texts = page.get("rec_texts", [])
        for i in range(len(texts)):
            txt = str(
                texts[i][0] if isinstance(texts[i], (list, tuple)) else texts[i]
            ).strip()
            y_mid = sum(p[1] for p in boxes[i]) / 4 / h
            blocks.append({"text": txt, "y": y_mid})

    blocks.sort(key=lambda b: b["y"])

    # --- 1. MRZ & CPR SCAN ---
    for b in blocks:
        txt = b["text"].replace(" ", "")
        # CPR
        cpr_match = re.search(r"(\d{9})", b["text"])
        if cpr_match and not fields["cpr"]:
            val = cpr_match.group(1)
            fields["cpr"] = val
            fields["cpr_verified"] = validate_bahrain_cpr(val)

        # MRZ (English Name & DOB)
        if "<<" in txt:
            # DOB
            m_dob = re.search(r"(\d{6})\d[MF]\d{7}", txt)
            if m_dob:
                yy, mm, dd = (
                    m_dob.group(1)[:2],
                    m_dob.group(1)[2:4],
                    m_dob.group(1)[4:6],
                )
                fields["dob"] = f"{dd}/{mm}/{'19' if int(yy) > 30 else '20'}{yy}"

            # English Name (Surname<<GivenName)
            if (
                re.search(r"^[A-Z<]+$", txt)
                and "BHR" not in txt
                and not fields["english_name"]
            ):
                parts = txt.split("<<")
                if len(parts) >= 2:
                    surname = parts[0].replace("<", " ").strip()
                    given = parts[1].replace("<", " ").strip()
                    fields["english_name"] = f"{given} {surname}".upper()

    # --- 2. ARABIC NAME ANCHOR ---
    # Keywords found in the footer that we MUST ignore
    footer_junk = ["هيئة", "المعلومات", "الحكومة", "بنة", "المطومات"]

    # Find "Name / الاسم" Label
    anchor_idx = -1
    for i, b in enumerate(blocks):
        if ("الاسم" in b["text"] or "Name" in b["text"]) and b["y"] > 0.5:
            anchor_idx = i
            break

    # If anchor found, look at the lines immediately following it
    if anchor_idx != -1:
        for i in range(anchor_idx + 1, min(anchor_idx + 5, len(blocks))):
            txt = blocks[i]["text"]

            # Check if it's Arabic
            if re.search(r"[\u0600-\u06FF]", txt):
                # Check if it contains footer keywords
                if any(junk in txt for junk in footer_junk):
                    continue

                # If it's a long Arabic string and not "Nationality"
                if len(txt.split()) >= 3 and "الجنسية" not in txt:
                    if not fields["arabic_name"]:
                        fields["arabic_name"] = txt
                        break

    return fields


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    ocr = PaddleOCR(enable_mkldnn=False, lang="ar")
    final = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
    }

    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None:
            continue
        res = extract_data(ocr.predict(img), img.shape[1], img.shape[0])

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

    print(json.dumps(final, indent=4, ensure_ascii=False))
