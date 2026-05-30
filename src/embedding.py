from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from data_loader import load_all_documents
# When Python executes a file directly, it adds the file's directory (src) to the import path, not the project root.
#imports work differently
class EmbeddingPipeline:
    #constructor
    #ie, this runs automatically when u create an object.
    # all-MiniLM-L6-v2 is the default embedding model and is lightweight, fast, outputs 384-dimensional vectors and is commonly used in RAB projects.

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int=1000, chunk_overlap: int = 200):
        """
        Chunk 1: chars 0-1000
        Chunk 2: chars 800-1800
        Chunk 3: chars 1600-2600

        Why overlap?

        Without overlap:

        Chunk 1:
        Machine learning is a field

        Chunk 2:
        of artificial intelligence

        The sentence gets broken.

        With overlap:

        Chunk 1:
        Machine learning is a field

        Chunk 2:
        is a field of artificial intelligence

        Important context is preserved.

        """
        self.model = SentenceTransformer(model_name)
        self.model_name=model_name
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        #len function is used to measure chunk size using character count. len("Hello") ->5
        #separators are used to split the text at logical boundaries. The splitter will try to split at the first separator ("\n\n"), if not found, it will try the next ("\n"), then space and finally character by character if no other separator is found. This helps to keep chunks coherent and meaningful.
        #if after a split, the chunk is larger than chunk_size, it will continue to split it using the next separator until all chunks are within the specified chunk_size.
        splitter = RecursiveCharacterTextSplitter(length_function=len, separators=["\n\n", "\n", " ", ""], chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        # the above line tells the langchain how to split text.
        chunks=splitter.split_documents(documents)
        #split_documents() is not your function. It is a built-in method provided by LangChain's RecursiveCharacterTextSplitter.
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
        #chunks - list of langchain document objects, where each document contains a chunk of text and its associated metadata. Langchain preserves the metadata from the original document objects.

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:    #returns a numpy array of embeddings. each row represents one chunk.
        texts = [chunk.page_content for chunk in chunks]
        """
        Suppose:

        chunks = [
            Document(page_content="Machine Learning"),
            Document(page_content="Deep Learning")
        ]

        Then:

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        becomes:

        texts = [
            "Machine Learning",
            "Deep Learning"
        ]

        Why?

        Because the embedding model works on text strings, not Document objects.
                
        """
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        # shape is : (number of chunks, embedding dimension=384)
        return embeddings
    
if __name__ == "__main__":
    docs=load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks=emb_pipe.chunk_documents(docs)
    embeddings=emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings)>0 else None)