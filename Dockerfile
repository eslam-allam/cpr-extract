# Use the official pixi-base image
FROM ghcr.io/prefix-dev/pixi:0.67.0-noble

# Set up the working directory
WORKDIR /app

# Copy your pixi.toml and pixi.lock first to leverage Docker layer caching
COPY pixi.toml pixi.lock ./

# Install the environment based on your lockfile
# This will install ccache, paddlepaddle, etc., into the container
RUN pixi install && pixi clean

# Copy the rest of your application code
COPY . .

# Set environment variables for silence and performance
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_LOG_LEVEL=3 \
    GLOG_minloglevel=3 \
    PYTHONUNBUFFERED=1

# Pre-download models using 'pixi run' to use the correct environment
RUN pixi run python -c \
    "from paddleocr import PaddleOCR; PaddleOCR(enable_mkldnn=False, lang='ar', ocr_version='PP-OCRv5')"

# Use 'pixi run' as the entrypoint to ensure the environment is activated
ENTRYPOINT ["pixi", "run", "python", "detect.py"]
