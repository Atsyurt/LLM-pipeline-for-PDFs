# The code snippet you provided is setting up a Python class called `LanguageModel` that integrates
# various libraries and tools for natural language processing tasks. Here's a breakdown of what each
# import statement is doing:
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
from uuid import uuid4
import requests
import os


# The `State` class is a Python class that defines a data structure with fields for a question,
# context, and answer.
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# The `LanguageModel` class defines methods to initialize a language model, embeddings, load PDF data,
# build a Milvus database, create a graph pipeline for question answering, and provide a method to
# query the model for responses.
class LanguageModel:
    def __init__(self):
        # Set up streaming callback manager for token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.ip_adress = os.getenv("IP_ADDRESS")
        if self.ip_adress == None:
            self.ip_adress = "localhost"

        self.create_rag_pipeline()

    def init_model(self):
        """
        The `init_model` function downloads a model file if it does not exist and then initializes the
        LlamaCpp model with specific parameters.
        """
        # Define file path and file name
        model_dir = "src/model"
        model_file = "gemma-3-4b-it-q4_0.gguf"
        # model_file = "gemma-3-4b-it-q4_0.gguf"
        model_path = os.path.join(model_dir, model_file)

        # Check if the model file exists
        if not os.path.exists(model_path):
            print(f"Model not found in {model_path}. Downloading...")
            # Ensure the directory structure exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # URL of the model file
            url = "https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/gemma-3-4b-it-q4_0.gguf"

            # Place put Your Hugging Face API token
            api_token = ""
            assert api_token != "", (
                "API token cannot be an empty string! please add hugging face api_token.see readme file for more info"
            )
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
            model_path="./src/model/" + model_file,
            temperature=0.4,
            max_tokens=2000,
            n_ctx=2000,
            top_p=1,
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
        )
        print("LLM installed successfully,")

    def init_embeding_model(self):
        """
        The `init_embeding_model` function initializes a HuggingFaceEmbeddings model with a specific model
        name and cache folder.
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="Snowflake/snowflake-arctic-embed-s", cache_folder="./"
        )
        print("Embedding model installed successfully,")

    def init_pdf_load(self):
        """
        The `init_pdf_load` function loads a PDF file, splits it into chunks of text, assigns unique IDs to
        each chunk, and prints a success message.
        """
        loader = PyPDFLoader("./data/dr_voss_diary.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=200
        )
        self.doc_chunks = text_splitter.split_documents(docs)
        for d in self.doc_chunks:
            d.metadata["id"] = str(uuid4())
        print("PDF loaded successfully,")

    def init_milvus_db(self):
        """
        The `init_milvus_db` function initializes a connection to a Milvus database using a specified URI
        and stores PDF data with corresponding embeddings in a collection named "pdf_data".
        """
        # URI = "http://localhost:19530"
        URI = "http://" + str(self.ip_adress) + ":19530"
        # URI = "http://127.0.0.1:19530"
        self.vector_store_pdf_data = Milvus.from_documents(
            self.doc_chunks,
            self.embeddings,
            collection_name="pdf_data",
            connection_args={"uri": URI},
        )

        print("Milvus db built successfully,")

    def init_graph_states(self):
        """
        The `init_graph_states` function retrieves relevant information from PDF data based on a given
        question and generates a response using a language model.
        :return: The function `init_graph_states` defines two inner functions `retrieve` and `generate`
        that are used to retrieve specific page content based on a given question and generate a
        response using a language model. The `retrieve` function retrieves specific page content by
        performing a similarity search on the embeddings of the question and the PDF data. The
        `generate` function generates a response by creating prompts based on the question
        """

        def retrieve(state: State):
            embedding_vector = self.embeddings.embed_query(state["question"])
            results = self.vector_store_pdf_data.similarity_search_by_vector(
                embedding_vector, k=5
            )
            # results = vector_store_pdf_data.similarity_search(state["question"], k=2 )
            specific_page_content = ""
            for res in results:
                id = res.metadata["id"]

                # Step 1: Load the PDF file

                for doc in self.doc_chunks:
                    if doc.metadata["id"] == str(id):
                        if doc.page_content != None:
                            specific_page_content = (
                                specific_page_content + "\n\n" + doc.page_content
                            )
                            break
            # retrieved_docs = vector_store_pdf_data.similarity_search(state["question"],k=2)
            return {"context": specific_page_content}

        def generate(state: State):
            docs_content = state["context"]
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
        """
        The `create_rag_pipeline` function initializes various components such as the LLM model, PDFs,
        Milvus DB, and graph states to create a RAG pipeline successfully.
        """
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
        """
        The `start_rag_pipeline` function takes a query, invokes a graph with the query, prints and returns
        the answer from the response.

        :param query: The `query` parameter in the `start_rag_pipeline` method is the question or query that
        will be passed to the `graph` object for processing. This query will be used to generate a response
        which will then be printed and returned by the method
        :return: The "answer" key from the response dictionary is being returned.
        """
        response = self.graph.invoke({"question": query})
        print(response["answer"])
        return response["answer"]

    # The `create_prompt` method in the `LanguageModel` class is a function that generates a prompt
    # message for the language model. Here's a breakdown of what it does:
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
        """
        The `ask` function queries the LLM and returns a response based on the input question.

        :param question: The `ask` method takes a question as input, which is expected to be a string. This
        question is then used to query the LLM (Large Language Model) to generate a response. The response
        is returned as a string
        :type question: str
        :return: The `ask` method is returning the response obtained from the `start_rag_pipeline` method.
        """
        """Query the LLM and return a response"""
        response = self.start_rag_pipeline(question)
        # response = self.llm(question)
        # return response["choices"][0]["text"].strip()  # Extract the text output
        return response
