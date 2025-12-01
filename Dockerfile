# base
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS base
ENV PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 app

# dev
FROM base AS dev
WORKDIR /tmp

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    pytorch-lightning \
    torch \
    torchmetrics \
    lightning \
    lightning-utilities \
    aim \
    biopandas \
    python-dotenv \
    scipy \
    docktgrid

WORKDIR /src
COPY . .
USER app
CMD ["python", "train.py", "--help"]

# prod
FROM base AS prod

RUN pip install --no-cache-dir docktdeep==0.1.1
USER app
WORKDIR /data
COPY ckpts /ckpts

ENTRYPOINT ["docktdeep", "predict", "--model-checkpoint", "/ckpts/docktdeep-model.ckpt"]
CMD ["--help"]