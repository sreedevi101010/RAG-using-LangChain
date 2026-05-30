import os # used for filesystem operations like creating directories, checking if files exist etc.
import faiss # Facebook AI Similarity Search - a library for efficient similarity search and clustering of dense vectors.
import numpy as np
import pickle # used for serializing and deserializing Python objects, such as saving and loading the FAISS index and metadata. pickling- converts the object into binary data, unpickling- converts the binary data back into the original object. it is stored in disk.
# large models can be pickled and saved to disk, and later loaded back into memory without needing to re-instantiate the model or re-compute the embeddings.
# here, used for storing metadata and the FAISS index, which can be large and time-consuming to compute. By pickling them, we can save the state of the vector store and quickly reload it when needed, without having to recompute everything from scratch.
from typing import List, Any
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingPipeline
from data_loader import load_all_documents

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", chunk_size: int=1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir # directory where faiss data is saved.
        os.makedirs(self.persist_dir, exist_ok=True) # ignore if the folder exists(keep it untouched)
        self.index = None # later -> self.index = faiss.IndexFlatL2(384) or self.IndexFlatIP(384)
        self.metadata = [] #faiss vector 0 has metadata[0] - maintained separately
       # self.embedding_model = embedding_model #stores the model name
        self.pipeline = EmbeddingPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # loads the model
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        print(f"[INFO] Loaded embedding model: {self.pipeline.model_name}")

    def build_from_documents(self, documents:List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents...")
        chunks=self.pipeline.chunk_documents(documents)
        embeddings=self.pipeline.embed_chunks(chunks)
        #metadatas=[{"text": chunk.page_content} for chunk in chunks] original metadat is not preserved here
        metadatas = []
        for chunk in chunks:
            metadata = chunk.metadata.copy()  # Start with the original metadata
            metadata["text"] = chunk.page_content  # Add the chunk text to the metadata
            metadatas.append(metadata)

        """
        [
            {
                "text": "Machine Learning"
            },
            {
                "text": "Deep Learning"
            }
        ]
        faiss only stores vectors. Everything else associated with a vector is typically called metadata, even if that "metadata" includes the chunk text.
        """
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        dim = embeddings.shape[1] # dimension of the embedding vectors, which is 384 for all-MiniLM-L6-v2, shape[0] = no. of chunks and shape[1] = 384. ie, (n, 384)
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim) # L2 distance for similarity search. For cosine similarity, use IndexFlatIP (Inner Product)
        self.index.add(embeddings) # add the new embeddings to the index
        if metadatas:
            self.metadata.extend(metadatas) # add the new metadata to the existing list of metadata. The order of metadata should correspond to the order of embeddings in the index. This way, when we retrieve a vector from the index, we can easily access its associated metadata using the same index position.   
        print(f"[INFO] Added {embeddings.shape[0]} vectors to the Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index") # the path persist_dir/faiss.index is built using os.path.join to ensure it works across different operating systems. This file will store the FAISS index, which contains the vector embeddings and their structure for efficient similarity search.
        meta_path = os.path.join(self.persist_dir, "metadata.pkl") # the path persist_dir/metadata.pkl is built using os.path.join to ensure it works across different operating systems. This file will store the metadata associated with the vectors in the FAISS index.
        faiss.write_index(self.index, faiss_path) # saves the FAISS index to disk. The index contains the vector embeddings and their structure for efficient similarity search.
        with open(meta_path, "wb") as f: # opens the metadata file in binary write mode., wb= write binary
            pickle.dump(self.metadata, f) # saves the metadata list to disk using pickle. This allows us to preserve the association between vectors and their metadata when we load the index later.
        print(f"[INFO] Faiss index and metadata saved to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path) # loads the FAISS index from disk. This allows us to restore the vector embeddings and their structure for similarity search.
        with open(meta_path, "rb") as f: # opens the metadata file in binary read mode, rb= read binary
            self.metadata = pickle.load(f) # loads the metadata list from disk using pickle. This restores the association between vectors and their metadata.
        print(f"[INFO] Faiss index and metadata loaded from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k:int =5):
        D, I = self.index.search(query_embedding, top_k) # D contains the distances of the nearest neighbors, and I contains their corresponding indices in the index. The query_embedding is converted to float32 to match the data type of the embeddings in the index.
        #I[0] = [[1, 0, 3]]
        #D[0] = [[0.12, 0.35, 1.82]]  in the order of closest to farthest.
        # they are 2d arrays because FAISS can handle batch queries. Even if we search with a single query, the results are returned in a 2D format where the first dimension corresponds to the number of queries (in this case, 1) and the second dimension corresponds to the number of nearest neighbors returned (top_k). ie, (1, top_k)
        results = []
        """
        zip makes the results like this:
        [
            (1, 0.12),
            (0, 0.35),
            (3, 1.82)
        ]
        """
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx<len(self.metadata) else {} # retrieve the metadata for the retrieved vector using its index. If the index is out of bounds (which shouldn't happen if everything is consistent), we return an empty dictionary.
            results.append({"index": idx, "metadata": meta, "distance": dist}) # we return the metadata and
        return results
    
    def query(self, query_text: str, top_k: int=5):
        print(f"[INFO] Querying vector store with: '{query_text}'")
        # query is usually small, so no chunking.
        query_embedding = self.pipeline.model.encode([query_text], show_progress_bar=False).astype('float32') # generate the embedding for the query text using the same embedding model. The query text
        return self.search(query_embedding, top_k=top_k)
    
if __name__ == "__main__":
    docs=load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is cognizant?", top_k=3))