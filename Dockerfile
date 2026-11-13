FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir \
      'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu' \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-api.txt

COPY helper_lib ./helper_lib
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
