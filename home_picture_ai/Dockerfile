# Home picture AI: a small Python service with ONNX Runtime. Every package is a ready-made wheel (nothing compiled
# here), at a fixed version, so the Pi builds the same thing each time.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

RUN pip install --only-binary=:all: \
        onnxruntime==1.31.1 numpy==2.5.5 pillow==13.0.0 zeroconf==0.154.0 \
        flatbuffers==25.12.19 packaging==26.3 protobuf==7.36.2 ifaddr==0.2.0

# Reading logos (1.2.0): RapidOCR's package only for its two PP-OCRv4 models (the server checks them; its own code,
# which would need OpenCV, isn't used).
RUN pip install --only-binary=:all: --no-deps rapidocr-onnxruntime==1.4.4

# The bench's 8-bit copies (1.5.1): ONNX Runtime's quantizer needs the ONNX package.
RUN pip install --only-binary=:all: --no-deps onnx==1.23.2 ml_dtypes==0.7.0 typing_extensions==4.16.0

COPY server.py direction.bin /app/

CMD ["python3", "-u", "/app/server.py"]
