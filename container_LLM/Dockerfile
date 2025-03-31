FROM python:3.11-slim

RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app.py
EXPOSE 9000
CMD ["gunicorn", "--bind", "0.0.0.0:9000", "app:app"]


