## Stage 1
FROM python:3.11-slim-bullseye as create_venv

# Install build dependencies
RUN apt-get update && apt-get install --no-install-recommends --yes build-essential

# Manage options
ARG EXTRAS_REQUIRE
RUN bash -c 'if [[ $EXTRAS_REQUIRE == *"dlib"* ]]; then apt-get install --no-install-recommends --yes cmake libopenblas-dev liblapack-dev; fi'

# Create venv and activate it
ENV VENV=/opt/venv
RUN python -m venv ${VENV}
ENV PATH="${VENV}/bin:${PATH}"

# Download DFX Extraction Library wheel
ADD https://s3.us-east-2.amazonaws.com/nuralogix-assets/dfx-sdk/python/libdfx-4.13.4-py3-none-linux_x86_64.whl /wheel/

# Add any local wheels
ADD wheels/* /wheel/

# Copy project files
WORKDIR /app
COPY *.py ./
COPY dfxdemo/*.py dfxdemo/
COPY dfxutils/*.py dfxutils/

# Switch to headless opencv
RUN sed -i "s/opencv-python/opencv-python-headless/" setup.py

# Install everything into the venv
ARG EXTRA_PYPI=""
RUN pip install wheel --no-cache-dir --disable-pip-version-check && \
    pip install .[${EXTRAS_REQUIRE}] --disable-pip-version-check --no-cache-dir ${EXTRA_PYPI:+--extra-index-url="$EXTRA_PYPI"} --find-links /wheel/

# Switch to headless opencv-contrib
RUN bash -c 'if [[ ${EXTRAS_REQUIRE} == *"mediapipe"* ]]; then pip uninstall -y opencv-python-headless opencv-contrib-python && pip install opencv-contrib-python-headless; fi'

## Stage 2
FROM python:3.11-slim-bullseye

# Install run dependencies
RUN apt-get update && apt-get install --no-install-recommends --yes libatomic1

# Manage options
ARG EXTRAS_REQUIRE
RUN bash -c 'if [[ ${EXTRAS_REQUIRE} == *"dlib"* ]]; then apt-get install --no-install-recommends --yes libopenblas0 liblapack3; fi'
RUN bash -c 'if [[ ${EXTRAS_REQUIRE} == *"visage"* ]]; then apt-get install --no-install-recommends --yes libgomp1; fi'
RUN bash -c 'if [[ ${EXTRAS_REQUIRE} == *"mediapipe"* ]]; then apt-get install --no-install-recommends --yes libgl1 libglib2.0-0; fi'

# Copy venv from previous stage and activate it
ENV VENV=/opt/venv
COPY --from=create_venv ${VENV} ${VENV}
ENV PATH="${VENV}/bin:${PATH}"
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/libvisage/lib:${LD_LIBRARY_PATH}

WORKDIR /app
ENTRYPOINT [ "dfxdemo" ]
