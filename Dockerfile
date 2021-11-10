FROM python:3.10-slim-bullseye as buildwheels
RUN apt-get update && apt-get install --no-install-recommends --yes \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    cmake
COPY *.py /app/
COPY dfxdemo/*.py /app/dfxdemo/
COPY dfxutils/*.py /app/dfxutils/
ADD https://s3.us-east-2.amazonaws.com/nuralogix-assets/dfx-sdk/python/libdfx-4.9.3.0-py3-none-linux_x86_64.whl /wheel/
RUN pip --disable-pip-version-check wheel -e /app --find-links /wheel -w /wheel

FROM python:3.10-slim-bullseye
RUN apt-get update && apt-get install --no-install-recommends --yes \
    libopenblas0 \
    liblapack3 \
    libgl1 \
    libglib2.0-0 \
    libatomic1
COPY --from=buildwheels /wheel /wheel
WORKDIR /app
RUN pip install --disable-pip-version-check --no-cache-dir --no-index --find-links /wheel dfxdemo
ENTRYPOINT [ "dfxdemo" ]