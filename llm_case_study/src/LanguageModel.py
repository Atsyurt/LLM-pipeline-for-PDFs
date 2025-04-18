from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate


class LanguageModel:
    def __init__(self, model_path):
        # Set up streaming callback manager for token-wise streaming
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Initialize LlamaCpp model
        self.llm = LlamaCpp(
            model_path=model_path,
            temperature=0.4,
            max_tokens=2000,
            n_ctx=2000,
            top_p=1,
            callback_manager=self.callback_manager,
            verbose=True,  # Verbose is required for callback manager
        )

    def ask(self, question: str) -> str:
        """Query the LLM and return a response"""
        # response = self.llm(question)
        # return response["choices"][0]["text"].strip()  # Extract the text output
        return "sa"
