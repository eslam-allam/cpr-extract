import re

# A comprehensive map of Demonyms to ISO-3 codes
# This covers major regions: Middle East, Asia, Europe, Africa, and the Americas.
NATIONALITY_MAP = {
    # --- GCC & Middle East ---
    "BAHRAINI": "BHR",
    "SAUDI": "SAU",
    "EMIRATI": "ARE",
    "KUWAITI": "KWT",
    "OMANI": "OMN",
    "QATARI": "QAT",
    "EGYPTIAN": "EGY",
    "JORDANIAN": "JOR",
    "LEBANESE": "LBN",
    "SYRIAN": "SYR",
    "IRAQI": "IRQ",
    "PALESTINIAN": "PSE",
    "YEMENI": "YEM",
    "SUDANESE": "SDN",
    "LIBYAN": "LBY",
    "TUNISIAN": "TUN",
    "ALGERIAN": "DZA",
    "MOROCCAN": "MAR",
    "TURKISH": "TUR",
    "IRANIAN": "IRN",
    # --- Asia & Oceania ---
    "INDIAN": "IND",
    "PAKISTANI": "PAK",
    "BANGLADESHI": "BGD",
    "FILIPINO": "PHL",
    "THAI": "THA",
    "SRI LANKAN": "LKA",
    "NEPALESE": "NPL",
    "CHINESE": "CHN",
    "JAPANESE": "JPN",
    "KOREAN": "KOR",
    "VIETNAMESE": "VNM",
    "MALAYSIAN": "MYS",
    "INDONESIAN": "IDN",
    "SINGAPOREAN": "SGP",
    "AUSTRALIAN": "AUS",
    "NEW ZEALANDER": "NZL",
    # --- Europe ---
    "BRITISH": "GBR",
    "FRENCH": "FRA",
    "GERMAN": "DEU",
    "ITALIAN": "ITA",
    "SPANISH": "ESP",
    "DUTCH": "NLD",
    "RUSSIAN": "RUS",
    "UKRAINIAN": "UKR",
    "PORTUGUESE": "PRT",
    "SWISS": "CHE",
    "SWEDISH": "SWE",
    "NORWEGIAN": "NOR",
    "IRISH": "IRL",
    "GREEK": "GRC",
    "POLISH": "POL",
    "ROMANIAN": "ROU",
    # --- Americas ---
    "AMERICAN": "USA",
    "CANADIAN": "CAN",
    "MEXICAN": "MEX",
    "BRAZILIAN": "BRA",
    "ARGENTINE": "ARG",
    "COLOMBIAN": "COL",
    "CHILEAN": "CHL",
    "PERUVIAN": "PER",
    # --- Africa ---
    "NIGERIAN": "NGA",
    "ETHIOPIAN": "ETH",
    "SOUTH AFRICAN": "ZAF",
    "KENYAN": "KEN",
    "GHANAIAN": "GHA",
    "UGANDAN": "UGA",
    "CAMEROONIAN": "CMR",
}

VALID_ISO_3 = set(NATIONALITY_MAP.values())


def validate_bahrain_cpr(cpr_str):
    """
    Validates Bahraini CPR using the official Weight-Position algorithm.
    Weights: 1, 2, 3, 4, 5, 6, 7, 8
    Logic: Sum % 11
    """
    if not cpr_str:
        return False

    # Ensure we only have the 9 digits
    clean_cpr = re.sub(r"\D", "", cpr_str)
    if len(clean_cpr) != 9:
        return False

    digits = [int(d) for d in clean_cpr]

    # Bahrain specific weights
    weights = [1, 2, 3, 4, 5, 6, 7, 8]

    # Calculate weighted sum of the first 8 digits
    total_sum = sum(digits[i] * weights[i] for i in range(8))

    # The check digit is the remainder of (Sum / 11)
    check_digit = total_sum % 11

    # Handle the case where remainder is 10 (rarely issued, but usually 0)
    if check_digit == 10:
        check_digit = 0

    return check_digit == digits[8]


def extract_data(results, h):
    fields = {
        "cpr": None,
        "cpr_verified": False,
        "arabic_name": None,
        "english_name": None,
        "dob": None,
        "nationality": None,
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

    for b in blocks:
        txt = b["text"].replace(" ", "")

        # 1. CPR Scan
        cpr_match = re.search(r"(\d{9})", b["text"])
        if cpr_match and not fields["cpr"]:
            val = cpr_match.group(1)
            fields["cpr"] = val
            fields["cpr_verified"] = validate_bahrain_cpr(val)

        # 2. MRZ Scan (Reliable 3-letter code)
        if "<<" in txt:
            m_mrz_data = re.search(r"(\d{6})\d[MF]\d{7}([A-Z]{3})", txt)
            if m_mrz_data:
                yy, mm, dd = (
                    m_mrz_data.group(1)[:2],
                    m_mrz_data.group(1)[2:4],
                    m_mrz_data.group(1)[4:6],
                )
                fields["dob"] = f"{dd}/{mm}/{'19' if int(yy) > 30 else '20'}{yy}"

                nat_code = m_mrz_data.group(2)
                if nat_code in VALID_ISO_3:
                    fields["nationality"] = nat_code

            # English Name
            if (
                re.search(r"^[A-Z<]+$", txt)
                and "BHR" not in txt
                and not fields["english_name"]
            ):
                parts = txt.split("<<")
                if len(parts) >= 2:
                    surname, given = (
                        parts[0].replace("<", " ").strip(),
                        parts[1].replace("<", " ").strip(),
                    )
                    fields["english_name"] = f"{given} {surname}".upper()

    # --- 3. FRONT CARD ANCHORS (Arabic Name & Full Nationality) ---
    footer_junk = ["هيئة", "المعلومات", "الحكومة", "بنة", "المطومات"]

    for i, b in enumerate(blocks):
        # Arabic Name
        if (
            ("الاسم" in b["text"] or "Name" in b["text"])
            and b["y"] > 0.5
            and not fields["arabic_name"]
        ):
            for j in range(i + 1, min(i + 5, len(blocks))):
                line = blocks[j]["text"]
                if re.search(r"[\u0600-\u06FF]", line) and not any(
                    junk in line for junk in footer_junk
                ):
                    if len(line.split()) >= 3 and "الجنسية" not in line:
                        fields["arabic_name"] = line
                        break

        # Nationality (Full Name Extraction)
        if ("الجنسية" in b["text"] or "Nationality" in b["text"]) and not fields[
            "nationality"
        ]:
            for j in range(i + 1, min(i + 3, len(blocks))):
                line_raw = blocks[j]["text"].strip().upper()
                # Check if any key in our map exists in the OCR line
                for nat_full, iso_code in NATIONALITY_MAP.items():
                    if nat_full in line_raw:
                        fields["nationality"] = iso_code
                        break
                if fields["nationality"]:
                    break

    return fields
