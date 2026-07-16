ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG CUDA_DOCKER_ARCH=70
ARG TURBOQUANT_REPO=https://github.com/turbo-tan/llama.cpp-tq3.git
ARG TURBOQUANT_REF=ed6eef39b2f934395c99b25f302d3272217e56ec

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG CUDA_DOCKER_ARCH
ARG TURBOQUANT_REPO
ARG TURBOQUANT_REF

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl gcc-14 g++-14 git \
        libgomp1 libssl-dev python3 && \
    rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 \
    CXX=g++-14 \
    CUDAHOSTCXX=g++-14

WORKDIR /src

RUN git clone --depth 1 "${TURBOQUANT_REPO}" llama.cpp && \
    cd llama.cpp && \
    git fetch --depth 1 origin "${TURBOQUANT_REF}" && \
    git checkout "${TURBOQUANT_REF}"

WORKDIR /src/llama.cpp

COPY qwen-think-fallback.patch /tmp/qwen-think-fallback.patch
RUN git apply /tmp/qwen-think-fallback.patch

RUN cmake -B build \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release --target llama-server -j"$(nproc)" && \
    mkdir -p /out && \
    cp build/bin/llama-server /out/ && \
    find build -name "*.so*" -exec cp -P {} /out/ \;

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build /out/ /app/
COPY entrypoint.sh /app/entrypoint.sh

ENV LD_LIBRARY_PATH=/app \
    LLAMA_ARG_HOST=0.0.0.0

WORKDIR /app

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8080/health >/dev/null || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
