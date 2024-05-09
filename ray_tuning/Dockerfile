# syntax=docker/dockerfile:1

FROM rayproject/ray:latest

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt


# What the container should run when it is started.
CMD ["python", "tune.py"]
