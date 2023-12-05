FROM python:3.9-slim
WORKDIR /app
COPY requirements requirements
COPY data data
COPY . .
RUN pip install -r requirements/dev.txt
CMD ["python", "repro.py"]