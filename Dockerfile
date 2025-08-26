FROM python:3.12-slim
ARG task_type

ENV TASK_TYPE=$task_type
ENV EXECUTE_IN_DOCKER=1

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r requirements.txt

COPY --chown=algorithm:algorithm run.py /opt/algorithm/
COPY --chown=algorithm:algorithm download.py /opt/algorithm/
COPY --chown=algorithm:algorithm KonfAI/UNetpp.py /opt/algorithm/UNetpp.py
COPY --chown=algorithm:algorithm KonfAI/UnNormalize.py /opt/algorithm/UnNormalize.py

RUN python download.py

ENTRYPOINT python -m run $0 $@
