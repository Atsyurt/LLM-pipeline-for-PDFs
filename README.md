# LLM-pipeline-for-PDFs
In this task, i built a simple pipeline to process and chunk a PDF document, store the chunks in a Milvus vector database, and implement a RAG system to answer questions about the document using an LLM

### 1. **Installation & Setup Instructions**

- **Environment Setup:**  
step1-)create a env for this setup using python venv
pip install virtualenv
"""
python -m venv .venv
"""

- **Dependency Installation:**  
step2-) Install requirements.txt

step3-) This setup needs langchain llamacpp  extension in order install it please see the link it depends the os you used
https://python.langchain.com/docs/integrations/llms/llamacpp/



- **Running the Scripts & Application:**  
- step4-) start the Milvus server
''' 
scripts\standalone_embed.bat start
'''
- step5-) start the fast api server
docker build -t 14050111012/rag_pipeline_bluecloud .

- step6-) to build image 

docker run --cpus=6 -p 80:80  14050111012/rag_pipeline_bluecloud

- step7-) start the fast api server locally if you want
uvicorn app:app --reload --host 0.0.0.0 --port 80

### 2. **Technical Discussion:**  