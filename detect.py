import sys
import re
import json
import cv2
from paddleocr import PaddleOCR

# -----------------------------
# 1. Standard Bahraini Checksum
# -----------------------------
def validate_bahrain_cpr(cpr_str):
    if not cpr_str or not re.match(r"^\d{9}$", cpr_str):
        return False
    digits = [int(d) for d in cpr_str]
    weights = [7, 6, 5, 4, 3, 2, 7, 6]
    total = sum(digits[i] * weights[i] for i in range(8))
    check = (11 - (total % 11)) % 11
    if check == 10: check = 0
    return check == digits[8]

# -----------------------------
# 2. Strict Filter Logic
# -----------------------------
def is_official_footer(text):
    """Detects and blocks the government authority footer text."""
    junk = ["هيئة", "المعلومات", "الحكومة", "الإلكترونية", "AUTHORITY", "EGOVERNMENT"]
    upper_text = text.upper()
    return any(word in upper_text for word in junk)

def clean_code_junk(text):
    """Removes IDBHR/IDBHD and trailing numbers from names."""
    # Matches IDBHR or IDBHD followed by any digits
    text = re.sub(r'IDBH[RD]\d*', '', text, flags=re.IGNORECASE)
    # Remove Bahrain card labels if they got attached
    text = re.sub(r'KINGDOM OF BAHRAIN|IDENTITY CARD', '', text, flags=re.IGNORECASE)
    return text.strip()

# -----------------------------
# 3. Targeted Extraction
# -----------------------------
def extract_data(results, w, h):
    fields = {"cpr": None, "cpr_verified": False, "arabic_name": None, "english_name": None, "dob": None}
    
    lines = []
    for page in list(results):
        boxes = page.get("dt_polys", [])
        texts = page.get("rec_texts", [])
        for i in range(len(texts)):
            txt = str(texts[i][0] if isinstance(texts[i], (list, tuple)) else texts[i]).strip()
            box = boxes[i]
            y = sum(p[1] for p in box) / 4 / h
            x = sum(p[0] for p in box) / 4 / w
            lines.append({"text": txt, "y": y, "x": x})

    # Priority: Sort Top to Bottom
    lines.sort(key=lambda l: l['y'])

    for line in lines:
        text, y, x = line['text'], line['y'], line['x']

        # A. CPR Detection (9 digits)
        cpr_match = re.search(r"(\d{9})", text)
        if cpr_match:
            val = cpr_match.group(1)
            is_valid = validate_bahrain_cpr(val)
            if is_valid:
                fields["cpr"], fields["cpr_verified"] = val, True
            elif not fields["cpr"]: 
                fields["cpr"] = val

        # B. DOB Extraction (MRZ or Label)
        if "<<" in text:
            clean_mrz = re.sub(r'[^A-Z0-9<]', '', text)
            m_dob = re.search(r"(\d{6})\d[MF]\d{7}", clean_mrz)
            if m_dob:
                yy, mm, dd = m_dob.group(1)[:2], m_dob.group(1)[2:4], m_dob.group(1)[4:6]
                fields["dob"] = f"{dd}/{mm}/{'19' if int(yy) > 30 else '20'}{yy}"

        # C. NAME EXTRACTION (THE FIX)
        # On Bahraini IDs, names are in the middle-bottom (0.5 to 0.85)
        # The Authority footer is usually at the very bottom (> 0.88)
        if 0.50 < y < 0.85:
            if not is_official_footer(text):
                # Arabic Name
                if re.search(r'[\u0600-\u06FF]', text) and len(text.split()) >= 2:
                    if not fields["arabic_name"] and "الاسم" not in text:
                        fields["arabic_name"] = text
                
                # English Name
                elif re.search(r'[A-Z]', text.upper()) and len(text) > 3:
                    cleaned = clean_code_junk(text)
                    if len(cleaned.split()) >= 2 and "<<" not in cleaned:
                        if not fields["english_name"]:
                            fields["english_name"] = cleaned.upper()

    # D. MRZ Backup for English Name (If front was unreadable)
    if not fields["english_name"]:
        for line in lines:
            if "<<" in line['text'] and re.search(r'[A-Z]', line['text']):
                parts = [p for p in line['text'].split('<<') if p]
                if len(parts) >= 2:
                    fields["english_name"] = f"{parts[1].replace('<', ' ')} {parts[0]}".strip().upper()

    return fields

# -----------------------------
# 4. Main execution
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    
    ocr = PaddleOCR(enable_mkldnn=False, lang='ar')
    final = {"cpr": None, "cpr_verified": False, "arabic_name": None, "english_name": None, "dob": None}

    for path in sys.argv[1:]:
        img = cv2.imread(path)
        if img is None: continue
        res = extract_data(ocr.predict(img), img.shape[1], img.shape[0])
        
        # Merge results (Take verified CPR over non-verified)
        if res["cpr_verified"]:
            final["cpr"], final["cpr_verified"] = res["cpr"], True
        elif not final["cpr"]: 
            final["cpr"] = res["cpr"]
        
        if res["arabic_name"]: final["arabic_name"] = res["arabic_name"]
        if res["english_name"]: final["english_name"] = res["english_name"]
        if res["dob"]: final["dob"] = res["dob"]

    print(json.dumps({"success": True, "data": final}, indent=4, ensure_ascii=False))
