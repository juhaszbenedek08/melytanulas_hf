FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY ./floating_requirements.txt requirements.txt

RUN pip install -U pip
RUN pip install -r requirements.txt