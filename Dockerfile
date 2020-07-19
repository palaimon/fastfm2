FROM python:3.7-slim

RUN apt-get update &&\
    apt-get install build-essential cmake -y && \
    apt-get clean

ENV POETRY_VERSION=1.0.5             \
    PYTHONUNBUFFERED=1               \
    PYTHONDONTWRITEBYTECODE=1        \
    PIP_NO_CACHE_DIR=off             \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100          \
    POETRY_NO_INTERACTION=1

RUN pip install poetry==$POETRY_VERSION && \
    poetry config virtualenvs.create false

COPY . /app

WORKDIR /app

RUN make

#temporary workaround
RUN poetry run python setup.py build_ext --inplace

#ENTRYPOINT /bin/sh
ENTRYPOINT poetry run pytest fastFM/tests -s -vvv