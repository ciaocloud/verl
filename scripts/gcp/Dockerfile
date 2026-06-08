ARG BASE_IMAGE=verlai/verl:vllm012.latest
FROM ${BASE_IMAGE}

WORKDIR /workspace/verl

COPY . /workspace/verl

RUN python3 -m pip install --no-deps -e /workspace/verl \
    && python3 -m pip install google-cloud-storage 'google-cloud-aiplatform[tensorboard]' tensorflow tensorboard

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENTRYPOINT ["/bin/bash", "/workspace/verl/scripts/gcp/train.sh"]
