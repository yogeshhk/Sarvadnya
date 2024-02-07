FROM python:3.11
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . /app
WORKDIR /app/
CMD ["chainlit","run","chainlit_main.py"]