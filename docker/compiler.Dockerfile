# Edge TPU compiler image.
#
# Google only ships `edgetpu_compiler` for x86-64 Linux, and the old Coral apt repository
# (packages.cloud.google.com/apt coral-edgetpu-stable) is no longer reachable. The compiler is
# still published as a self-contained bundle (own loader + libc) in github.com/google-coral/edgetpu,
# so we fetch it from there. On Apple Silicon the image runs under emulation; compiling a Reef
# model takes a few seconds.
#
#   docker build --platform linux/amd64 -f docker/compiler.Dockerfile -t reef-edgetpu-compiler docker/
#   docker run --rm --platform linux/amd64 -v "$PWD":/work reef-edgetpu-compiler \
#       edgetpu_compiler -s -o /work/artifacts /work/artifacts/reef_int8.tflite
FROM --platform=linux/amd64 debian:bookworm-slim

ARG EDGETPU_REF=master
ARG BASE=https://raw.githubusercontent.com/google-coral/edgetpu/${EDGETPU_REF}/compiler/x86_64

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/edgetpu/edgetpu_compiler_bin \
 && cd /opt/edgetpu \
 && curl -fsSLO ${BASE}/edgetpu_compiler \
 && cd edgetpu_compiler_bin \
 && for f in edgetpu_compiler ld-linux-x86-64.so.2 libc++.so.1 libc++abi.so.1 libc.so.6 libdl.so.2 \
             libm.so.6 libpthread.so.0 libresolv.so.2 librt.so.1 LICENSES NOTICE; do \
      curl -fsSLO ${BASE}/edgetpu_compiler_bin/$f; \
    done \
 && chmod +x /opt/edgetpu/edgetpu_compiler /opt/edgetpu/edgetpu_compiler_bin/edgetpu_compiler \
             /opt/edgetpu/edgetpu_compiler_bin/ld-linux-x86-64.so.2

# The wrapper resolves its bundle relative to its own location, so put it on PATH (no symlink).
ENV PATH=/opt/edgetpu:$PATH
WORKDIR /work
CMD ["edgetpu_compiler", "--version"]
