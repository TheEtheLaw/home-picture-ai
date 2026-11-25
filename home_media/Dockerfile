# Home media: a small Python service. The drive's tools come from Debian (lsblk, wipefs, mount, mkfs.ext4); the one
# Python package (for being found on the network) is a ready-made wheel at a fixed version, as in Home picture AI.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

RUN apt-get update \
    && apt-get install -y --no-install-recommends util-linux mount e2fsprogs \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --only-binary=:all: zeroconf==0.154.0 ifaddr==0.2.0

COPY server.py /app/

CMD ["python3", "-u", "/app/server.py"]
