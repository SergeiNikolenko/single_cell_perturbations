FROM python:3.9-slim
WORKDIR /app
COPY requirements requirements
COPY data data
COPY config config
COPY . .
RUN pip install -r requirements/dev.txt
CMD ["python", "repro.py"]