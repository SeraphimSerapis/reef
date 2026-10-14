# Reef runtime image: `reef serve` with the Edge TPU runtime, for Linux amd64 and arm64.
#
#   docker build -f docker/Dockerfile -t reef .
#   docker run --rm -p 8080:8080 --device /dev/bus/usb -v reef-data:/data reef          # model from the Hub
#   docker run --rm -p 8080:8080 --device /dev/bus/usb -v ./my-model:/model -e REEF_MODEL=/model reef
#
# USB Coral: pass /dev/bus/usb (or run privileged). M.2 / PCIe Coral: install the gasket/apex
# driver on the host and pass /dev/apex_0. Without a Coral the server falls back to the CPU.
FROM python:3.12-slim-bookworm

ARG TARGETARCH
# "std" runs the Edge TPU at reduced clock; "max" at full clock (runs hotter, see the Coral docs).
ARG LIBEDGETPU=std
ARG LIBEDGETPU_TAG=16.0TF2.19.1-1
ARG LIBEDGETPU_VER=16.0tf2.19.1-1

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl libusb-1.0-0 \
 && curl -fsSL -o /tmp/libedgetpu.deb \
      "https://github.com/feranick/libedgetpu/releases/download/${LIBEDGETPU_TAG}/libedgetpu1-${LIBEDGETPU}_${LIBEDGETPU_VER}.bookworm_${TARGETARCH}.deb" \
 && echo "libedgetpu1-max libedgetpu/accepted-eula boolean true" | debconf-set-selections \
 && apt-get install -y --no-install-recommends /tmp/libedgetpu.deb \
 && rm -rf /tmp/libedgetpu.deb /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install . \
 && useradd --system --uid 1000 --create-home reef \
 && mkdir -p /data && chown reef /data
USER reef

# The model: a Hugging Face repo id (downloaded once into /data) or a mounted directory like /model.
ENV REEF_MODEL=seraphimserapis/reef HF_HOME=/data/hf
VOLUME /data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz')"
ENTRYPOINT ["reef"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
