FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt update && apt install -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# env for experiments
RUN pip install \
  "datasets == 3.6.0" \
  "pandas" \
  "protobuf" \
  "tiktoken" \
  "blobfile" \
  "sentencepiece == 0.2.0" \
  "matplotlib" \
  "outlines == 1.1.1" \
  "xgrammar == 0.1.24" \
  "syncode == 0.4.12"

COPY truncproof/ /opt/TruncProof/truncproof/
COPY pyproject.toml /opt/TruncProof/
COPY README.md /opt/TruncProof/
RUN pip install /opt/TruncProof

WORKDIR /workspace
