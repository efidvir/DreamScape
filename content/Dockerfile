FROM python:3.9-slim

# Install build tools
RUN apt-get update && apt-get install -y gcc build-essential
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set the environment variable so that Flask (or your web framework) listens on 0.0.0.0
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["python", "app.py"]
