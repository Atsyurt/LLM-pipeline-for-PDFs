# LLM-pipeline-for-PDFs
In this task, i built a simple pipeline to process and chunk a PDF document, store the chunks in a Milvus vector database, and implement a RAG system to answer questions about the document using an LLM

### 1. **Installation & Setup Instructions**

- **Environment Setup:**

# Note: Make sure that your python version is 3.11

* step1-)create a env for this setup using python venv

    `pip install virtualenv`

    `python -m venv .venv`


- **Dependency Installation:**  
* step2-) Install requirements.txt

    `pip install -r requirements.txt`

step3-) To ensure that the system doesn't require high hardware resources when running locally, this application uses LLM inference through the langchain llama.cpp extension. To install it, please refer to the link, which depends on the operating system you are using.

https://python.langchain.com/docs/integrations/llms/llamacpp/


- **Running the Scripts & Application:**  
* step4-) start the Milvus server

    `scripts\standalone_embed.bat start`

* step5-) start the fast api server locally

    `uvicorn app:app --reload --host 0.0.0.0 --port 80`

* step6-) Build fast api server with docker if you want

    `docker build -t {custom_image_name} .`

* step7-) Start the fast api with Docker if you want

    `docker run --cpus=6 -p 80:80  14050111012/rag_pipeline_bluecloud`

* step8-) In order to see complete evaluation please run eval.py

    `python eval.py`


### 2. **Technical Discussion:**  