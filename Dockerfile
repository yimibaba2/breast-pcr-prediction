# 乳腺癌 pCR 预测系统 - 可复现部署镜像
FROM python:3.11-slim

WORKDIR /app

# 编译 pyradiomics 所需系统依赖 + matplotlib/scikit-image 运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先锚定 numpy 1.26（pyradiomics 仅兼容 numpy 1.x）
RUN pip install --no-cache-dir "numpy==1.26.4"

# 其余依赖（pip 自动选与 numpy 1.26 兼容的版本）
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 编译安装 pyradiomics（用已装好的 numpy 1.26 头文件）
RUN pip install --no-cache-dir versioneer
RUN pip install --no-cache-dir "pyradiomics==3.0.1" --no-build-isolation

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/

WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
