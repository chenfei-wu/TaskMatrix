FROM ubuntu:22.04

RUN apt-get -y update
RUN apt-get install -y git python3.11 python3-pip supervisor
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN ln -s /usr/bin/python3 /usr/bin/python
COPY src/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY src /app/src

WORKDIR /app/src
ENV WORKERS 2
CMD supervisord -c supervisord.conf
