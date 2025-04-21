FROM python:3.11
WORKDIR /usr/local/app


#Please change this

ENV IP_ADDRESS 192.168.2.50


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn
#Install llama cpp
RUN pip install llama-cpp-python
RUN mkdir src
RUN mkdir data

# Copy in the source code
COPY ./app.py ./
COPY ./src /usr/local/app/src
COPY ./data /usr/local/app/data
RUN ls

EXPOSE 5000
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
