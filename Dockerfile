# RetiVASC server for a Linux GPU host (NVIDIA driver + NVIDIA Container Toolkit on the host).
# The image holds the code, the built web UI and all model weights, so it runs without internet.
# Fetch the weights first (python tools/release.py fetch), then:
#   docker build -t retivasc:0.1 .
#   docker save retivasc:0.1 | gzip > retivasc-0.1.tar.gz      # carry into the intranet
#   docker compose -f deploy/docker-compose.yml up -d

FROM node:20-slim AS webui
WORKDIR /webui
COPY webui/package.json webui/package-lock.json ./
RUN npm ci
COPY webui/ ./
RUN npx vite build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /app
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
COPY automorph_v2/requirements/ automorph_v2/requirements/
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt
COPY automorph_v2/ automorph_v2/
COPY retivasc_server/ retivasc_server/
COPY --from=webui /webui/dist webui/dist
COPY webui/public/samples webui/public/samples
COPY deploy/retivasc.local.json ./retivasc.local.json
VOLUME /data
EXPOSE 8080
CMD ["python", "-m", "retivasc_server", "serve"]
