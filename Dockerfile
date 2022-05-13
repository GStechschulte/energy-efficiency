# select base image
FROM python:3.8

# Set working directory 
WORKDIR /model

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./config/config.ini .
COPY ./src ./gp_scripts

# ./scripts/__main__.py goes off of ./scripts above, i.e., the name
# given inside the container (not the GitHub repo directory)
CMD ["python", "./gp_scripts/__main__.py"]