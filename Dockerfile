FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime
RUN pip install --no-cache-dir docktdeep==0.1.1

RUN useradd -m -u 1000 app
USER app

WORKDIR /data
COPY ckpts /ckpts

ENTRYPOINT ["docktdeep", "predict", "--model-checkpoint", "/ckpts/docktdeep-model.ckpt"]
CMD ["--help"]