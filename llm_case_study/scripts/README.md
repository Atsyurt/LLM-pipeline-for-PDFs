

 **Note:**  All of my data preparation pipelines are included in
 the ../src/LanguageModel.py from here You can view the data preparation process from there.

 ```python
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
 ```
