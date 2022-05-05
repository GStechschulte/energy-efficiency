# select a base image
FROM python:3.8

# Set working directory 
RUN mkdir /model
WORKDIR /model

# Create conda environment from yaml file
COPY environment.yml  .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "env", "/bin/bash", "-c"]

# Activate environment
RUN conda activate thesis

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
#ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY /src /model

ENTRYPOINT ["python", "/src/__main__.py"]