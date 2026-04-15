import sys
import re
import json
import cv2
import contextlib
import os
import argparse


@contextlib.contextmanager
def suppress_stdout_stderr(disabled=False):
    """Deep redirect of stdout/stderr file descriptors to /dev/null."""
    if disabled:
        yield
        return
    # Open devnull
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Save original FDs
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    try:
        # Redirect FD 1 and 2 to devnull
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        # Restore original FDs
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        # Close temp FDs
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        os.close(devnull)


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
    parser = argparse.ArgumentParser(
        description="Extract and validate Bahraini CPR data from images."
    )

    # Positional argument for one or more image paths
    parser.add_argument(
        "images",
        nargs="+",
        help="Path to one or more image files (e.g., front.jpg back.jpg)",
    )

    # Optional flags
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show PaddleOCR logs)",
    )
    parser.add_argument(
        "-m",
        "--mkldnn",
        action="store_true",
        help="Enable MKLDNN acceleration (CPU only)",
    )
    parser.add_argument(
        "-c",
        "--model-source-check",
        action="store_true",
        help="Enable model source check (verify host connectivity)",
    )

    args = parser.parse_args()

    with suppress_stdout_stderr(disabled=args.verbose):
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = str(
            not args.model_source_check
        )

        from paddleocr import PaddleOCR

        # Apply environment silencers even if verbose is off

        # Initialize OCR with the new mkldnn parameter
        ocr = PaddleOCR(enable_mkldnn=args.mkldnn, lang="ar")

        final = {
            "cpr": None,
            "cpr_verified": False,
            "arabic_name": None,
            "english_name": None,
            "dob": None,
        }

        for path in args.images:
            img = cv2.imread(path)
            if img is None:
                if args.verbose:
                    print(f"Warning: Could not read image at {path}", file=sys.stderr)
                continue

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

    # Final result is always clean JSON to stdout
    print(json.dumps(final, indent=4, ensure_ascii=False))
