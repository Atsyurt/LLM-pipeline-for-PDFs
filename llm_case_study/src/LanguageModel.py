from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader

import requests
import os


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class LanguageModel:
    def __init__(self):
        # Set up streaming callback manager for token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # # Initialize LlamaCpp model
        # self.llm = LlamaCpp(
        #     model_path=model_path,
        #     temperature=0.4,
        #     max_tokens=2000,
        #     n_ctx=2000,
        #     top_p=1,
        #     callback_manager=self.callback_manager,
        #     verbose=True,  # Verbose is required for callback manager
        # )
        self.create_rag_pipeline()

    def init_model(self):
        # Define file path and file name
        model_dir = "src/model"
        model_file = "gemma-3-4b-it-qat-q4_0-gguf"
        # model_file = "gemma-3-4b-it-q4_0.gguf"
        model_path = os.path.join(model_dir, model_file)

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Model not found in {model_path}. Downloading...")
            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # URL of the model file
            url = "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/gemma-3-4b-it-q4_0.gguf"
            # Your Hugging Face API token
            api_token = ""

            # Download the model file with authentication
            headers = {"Authorization": f"Bearer {api_token}"}
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code == 200:
                print("model is donwnloading now ...")
                with open(model_path, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Model successfully downloaded to {model_path}")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status_code}"
                )
        else:
            print(f"Model already exists at {model_path}.")

        self.llm = LlamaCpp(
            model_path="./src/model/gemma-3-4b-it-q4_0.gguf",
            temperature=0.4,
            max_tokens=2000,
            n_ctx=2000,
            top_p=1,
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        print("LLM installed successfully,")

    def init_embeding_model(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-s", cache_folder="./"
        )
        print("Embedding model installed successfully,")

    def init_pdf_load(self):
        loader = PyPDFLoader("./data/dr_voss_diary.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=500
        )
        self.doc_chunks = text_splitter.split_documents(docs)
        print("PDF loaded successfully,")

    def init_milvus_db(self):
        URI = "http://localhost:19530"
        self.vector_store_pdf_data = Milvus.from_documents(
            self.doc_chunks,
            self.embeddings,
            collection_name="pdf_data",
            connection_args={"uri": URI},
        )
        print("Milvus db built successfully,")

    def init_graph_states(self):
        def retrieve(state: State):
            embedding_vector = self.embeddings.embed_query(state["question"])
            results = self.vector_store_pdf_data.similarity_search_by_vector(
                embedding_vector, k=2
            )
            # retrieved_docs = self.vector_store_pdf_data.similarity_search(
            #     state["question"], k=2
            # )
            return {"context": results}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = self.create_prompt(state["question"], docs_content)
            print(messages)
            # messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = self.llm.invoke(messages)
            print("RESPONSE", response)
            return {"answer": response}

        self.graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        self.graph_builder.add_edge(START, "retrieve")
        self.graph = self.graph_builder.compile()
        print("rag graph pipeline built successfully,")

    def create_rag_pipeline(self):
        print("LLM model is loading..")
        self.init_model()
        self.init_embeding_model()
        print("PDF's are loading..")
        self.init_pdf_load()
        print("Milvus DB is loading..")
        self.init_milvus_db()
        print("Graph states are loading..")
        self.init_graph_states()
        print("RAG pipeline created successfully.")

    def start_rag_pipeline(self, query):
        response = self.graph.invoke({"question": query})
        print(response["answer"])
        return response["answer"]

    def create_prompt(
        self,
        task_description,
        context,
        system_prompt="You are my AI assistant, helping me get selected for the BlueCloud job. Your task is to answer all questions as logically, clearly, and concisely as possible, ensuring that your responses are well-structured and professional",
    ):
        context_instructions = """
        Answer the question directly based on the context below. 
        If the question cannot be answered using the information provided
        or if uncertainty exists, respond with 'I don't know
        """

        """Creates a prompt that includes the task description"""
        message = f"""<bos><start_of_turn>user
                    {system_prompt}\n{context_instructions}\n
                    Context:{context}
                    <end_of_turn>
                    <start_of_turn>user
                    {task_description}
                    <end_of_turn>
                    <start_of_turn>model"""

        # print("create_prompt", prompt)
        return message

    def ask(self, question: str) -> str:
        """Query the LLM and return a response"""
        response = self.start_rag_pipeline(question)
        # response = self.llm(question)
        # return response["choices"][0]["text"].strip()  # Extract the text output
        return response
