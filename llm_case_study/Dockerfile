FROM python:3.11
WORKDIR /usr/local/app


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
#Install llama cpp
RUN pip install llama-cpp-python
RUN mkdir src
# Copy in the source code
COPY ./app.py ./
COPY ./src /usr/local/app/src
RUN ls

EXPOSE 5000
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
