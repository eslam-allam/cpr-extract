from sanic import Sanic, response
import cv2
import numpy as np
import pickle
import worker_init

app = Sanic("OCR_Service")


@app.before_server_start
async def setup_ocr(app, loop):
    worker_init.initialize()


@app.get("/health")
async def health_check(request):
    ocr = worker_init.get_ocr()
    if ocr is not None:
        return response.json({"status": "healthy", "model_loaded": True}, status=200)
    return response.json({"status": "unhealthy", "model_loaded": False}, status=503)


@app.post("/predict")
async def predict_image(request):
    file = request.files.get("image")
    if not file:
        return response.raw(b"No image provided", status=400)

    nparr = np.frombuffer(file.body, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ocr = worker_init.get_ocr()
    raw_results = ocr.predict(img)

    # Safely convert raw_results to a list of pages
    try:
        pages = list(raw_results)
    except TypeError:
        pages = [raw_results]

    cleaned_results = []

    for page in pages:
        if not isinstance(page, dict):
            continue

        # 1. Extract and clean bounding boxes (dt_polys)
        boxes = page.get("dt_polys", [])
        cleaned_boxes = []
        for box in boxes:
            try:
                # Cast coordinates to pure float lists to strip any NumPy wrappers
                cleaned_box = [[float(pt[0]), float(pt[1])] for pt in box]
                cleaned_boxes.append(cleaned_box)
            except (IndexError, TypeError, ValueError):
                continue

        # 2. Extract and clean recognized texts (rec_texts)
        texts = page.get("rec_texts", [])
        cleaned_texts = []
        for txt in texts:
            try:
                if isinstance(txt, (list, tuple)):
                    # Format: ("text_string", confidence_score)
                    cleaned_texts.append((str(txt[0]), float(txt[1])))
                else:
                    cleaned_texts.append(str(txt))
            except (IndexError, TypeError, ValueError):
                continue

        cleaned_results.append({"dt_polys": cleaned_boxes, "rec_texts": cleaned_texts})

    payload = {"results": cleaned_results, "shape_0": img.shape[0]}

    binary_data = pickle.dumps(payload)
    return response.raw(binary_data, content_type="application/octet-stream")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, workers=1, access_log=False)
