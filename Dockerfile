FROM python:3.10-slim-bullseye

ENV USER=python
RUN groupadd --system ${USER} \
    && useradd --system --gid ${USER} ${USER}

ENV METRICS_PORT=8000
EXPOSE ${METRICS_PORT}

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 --disable-pip-version-check --no-cache-dir install -r requirements.txt

COPY . .

USER ${USER}

ENTRYPOINT ["python3", "analog_gauge_reader.py"]

CMD ["serve"]
