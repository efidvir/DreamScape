FROM python:3.9-slim

WORKDIR /app

# Copy and install Python dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend app code.
COPY app.py .

# Expose the port the Flask app will run on.
EXPOSE 5000

# Start the Flask app.
CMD ["python", "app.py"]
