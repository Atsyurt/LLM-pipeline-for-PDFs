# LLM-pipeline-for-PDFs
In this task, i built a simple pipeline to process and chunk a PDF document, store the chunks in a Milvus vector database, and implement a RAG system to answer questions about the document using an LLM

### 1. **Installation & Setup Instructions**

- **Environment Setup:**

# Note: Make sure that your python version is 3.11

* step1-)Create a env for this setup using python venv

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
* You should see similiar outputs for the accuracy score like this

![Alt text](llm_case_study/src/images/study_result.png)


### 2. **Technical Discussion:**

# High Overview of RAG pipeline 
![Alt text](llm_case_study/src/images/pipeline_high_level_overview.png)

- First, I want to talk about the RAG pipeline. The RAG pipeline I developed using Langchain, Llama.cpp, and LangGraph is fully customizable for different scenarios. As part of the case study, this pipeline focuses solely on retrieving content and generating answers based on the retrieved content. As seen in its node-based structure, the pipeline can be adapted to various scenarios with different nodes and logic. A more advanced RAG pipeline can be developed in this way. LangGraph provides us with all the necessary tools.

- As the language model, I used the Gemma3 series, specifically the Gemma3 4B inst-tuned model. I utilized this model in GGUF format, which is a quantized format compatible with Llama.cpp. This allows the model to function seamlessly on systems with low hardware resources, even within a RAG system, without requiring GPU resources.
I could have also used the Llama model with its GGUF format. While Llama is slightly more successful and performs a bit better compared to Gemma, the difference between the two isn't significant. I opted for Google's Gemma because I found its documentation and community support to be better.

- For the embedding model, I used the recommended Snowflake model. I also tested other models, but the performance and accuracy didn't vary significantly. Therefore, I chose the Snowflake-Arctic-Embed-S model

- In terms of success, chunk size and overlap size in information retrieval from the vector database had a significant impact on performance. High chunk sizes, between 800-1500, considerably reduced success. Given the limited amount of data, such high chunk sizes are not suitable for this use case. By setting the chunk size between 200-300 and the overlap size between 80-100, performance improved and yielded better results.

- The most effective strategy for this study was utilizing language models in GGUF format. This allowed the models to perform efficiently even on my laptop with low computational resources. In my opinion, using such optimization tools for local systems is absolutely critical.

- I prepared Docker files for the system, which can be used to turn the entire pipeline into a Docker image.However, network configurations need to be set up for communication with the Milvus database (Docker Compose can be used). Due to limited time, I only prepared the Docker files and left them as is. With minor connection adjustments, these images can be made to work quite easily.

