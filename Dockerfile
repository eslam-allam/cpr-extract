# Use the official pixi-base image
FROM ghcr.io/prefix-dev/pixi:0.67.0-noble

# Set up the working directory
WORKDIR /app

# Install system dependencies for OpenCV and OpenMP
# libgl1 is what provides libGL.so.1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Copy your pixi.toml and pixi.lock first to leverage Docker layer caching
COPY pixi.toml pixi.lock ./

# Install the environment based on your lockfile
# This will install ccache, paddlepaddle, etc., into the container
RUN pixi install --locked && pixi clean

# Set environment variables for silence and performance
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PADDLE_LOG_LEVEL=3 \
    GLOG_minloglevel=3 \
    PYTHONUNBUFFERED=1

# Pre-download models using 'pixi run' to use the correct environment
RUN pixi run python -c \
    "from paddleocr import PaddleOCR; PaddleOCR(enable_mkldnn=False, lang='ar', ocr_version='PP-OCRv5')"

# Copy the rest of your application code
COPY . .

# Use 'pixi run' as the entrypoint to ensure the environment is activated
ENTRYPOINT ["pixi", "run"]

CMD [ "python", "server.py" ]
