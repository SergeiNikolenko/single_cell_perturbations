FROM python:3.9-slim
WORKDIR /app

COPY requirements requirements
COPY data data
RUN ls
COPY . .
RUN pip install pip-tools

RUN pip-compile -v requirements/dev.in
RUN pip-sync requirements/dev.txt
CMD ["python", "repro.py"]