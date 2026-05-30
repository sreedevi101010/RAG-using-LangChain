import os
from dotenv import load_dotenv
from vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
#Chatgroq is used to access the llm model through Groq's API.
load_dotenv()
# python searches for .env file in the current working directory, which is src when we run python from src. Since our .env file is in the project root, we need to specify the path to it using load_dotenv("../.env"). This ensures that the environment variables defined in the .env file are loaded correctly, regardless of where we run the script from.


class RAGSearch:
    def __init__(self, persist_dir:str = "faiss_store", embedding_model:str = "all-MiniLM-L6-v2", llm_model: str="llama-3.3-70b-versatile"):
        self.vectorstore = FaissVectorStore(persist_dir=persist_dir, chunk_size=1000, chunk_overlap=200) # initializes the FaissVectorStore with the specified persist_dir, chunk_size, and chunk_overlap. This will load the embedding model and prepare the vector store for use.
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model=llm_model, api_key=groq_api_key)
        print(f"[INFO] Loaded LLM model: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts=[r["metadata"].get("text", "") for r in results if r["metadata"]]
        """
        Suppose:

        r["metadata"] = {
            "text": "Machine Learning..."
        }

        Then:

        r["metadata"].get("text", "")

        returns:

        "Machine Learning..."
                
        """
        #.get() is used because if text doesnt exist, then instead of system crashing, it can return "".
        context = "\n\n".join(texts)
        """
        Joins all chunks together.

        Suppose:

        texts = [
            "Chunk A",
            "Chunk B",
            "Chunk C"
        ]

        Then:

        context

        becomes:

        Chunk A

        Chunk B

        Chunk C

        The separator:

        "\n\n"

        adds blank lines between chunks.
        """
        if not context:
            return "No relevant documents found"
        prompt=f"""Summarize the following context for the query: '{query}'\n\nContext:{context}\n\nSummary:"""
        response = self.llm.invoke(prompt) #Sends prompt to Groq        
        return response.content
    

if __name__ == "__main__":
    rag_search=RAGSearch()
    query = "What is Machine Learning?"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)