ARG BASE_IMAGE=verlai/verl:vllm012.latest
FROM verlai/verl:vllm012.latest

WORKDIR /workspace

RUN python3 -m pip install --no-cache-dir \
        google-cloud-storage \
        'google-cloud-aiplatform[tensorboard]' \
        tensorflow \
        tensorboard

ARG GIT_REMOTE=https://github.com/ciaocloud/verl.git
RUN git clone "${GIT_REMOTE}" /workspace/verl \
    && python3 -m pip install --no-deps -e /workspace/verl

COPY lab/gcp/bootstrap.sh /workspace/bootstrap.sh
RUN chmod +x /workspace/bootstrap.sh

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/bin/bash", "/workspace/bootstrap.sh"]
