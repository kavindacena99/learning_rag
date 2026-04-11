import os
import numpy as np
from dotenv import load_dotenv
from google import genai

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001") # Latest standard for 2026
RAG_RESOURCE_PATH = os.getenv("RAG_RESOURCE_PATH", "resources/knowledge.txt")

def resolve_resource_path(file_path):
    if os.path.isabs(file_path):
        return file_path

    return os.path.join(PROJECT_ROOT, file_path)

class SemanticRAG:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.chunks = []
        self.embeddings = []

    def read_resource_text(self, file_path):
        resolved_path = resolve_resource_path(file_path)

        if not os.path.exists(resolved_path):
            print(f"Error: {resolved_path} not found.")
            return None

        file_extension = os.path.splitext(resolved_path)[1].lower()

        if file_extension == ".txt":
            with open(resolved_path, "r", encoding="utf-8") as f:
                return f.read()

        if file_extension == ".pdf":
            if PdfReader is None:
                print("Error: PDF support requires the 'pypdf' package.")
                print("Install it with: python -m pip install pypdf")
                return None

            try:
                reader = PdfReader(resolved_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages).strip()
            except Exception as e:
                print("Error: Failed to read PDF resource.")
                print(f"Error Details: {e}")
                return None

        print(f"Error: Unsupported resource type '{file_extension}'.")
        return None

    def load_and_chunk(self, file_path):
        """Reads text and splits into chunks."""
        text = self.read_resource_text(file_path)
        if not text:
            print("Error: No resource text could be loaded.")
            return False

        size = 800
        self.chunks = [text[i:i+size] for i in range(0, len(text), size)]
        total_chunks = len(self.chunks)
        print(f"Created {total_chunks} chunks. Generating embeddings...")

        # Process in batches of 100 to avoid the ClientError
        batch_size = 100
        self.embeddings = []

        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(total_chunks-1)//batch_size + 1}...")
            
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            # Add the new batch of embeddings to our main list
            self.embeddings.extend([e.values for e in result.embeddings])

        return True

    def semantic_search(self, query, top_k=3):
        """Finds most relevant chunks using cosine similarity."""
        # 1. Embed the user query
        query_result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_vec = np.array(query_result.embeddings[0].values)

        # 2. Calculate similarities (Dot Product on normalized vectors)
        scores = []
        for i, chunk_vec in enumerate(self.embeddings):
            chunk_vec = np.array(chunk_vec)
            # Cosine similarity formula
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(chunk_vec)
            score = np.dot(query_vec, chunk_vec) / (norm_q * norm_c)
            scores.append((score, self.chunks[i]))

        # 3. Sort by highest score
        scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scores[:top_k]]

    def generate_answer(self, query, context_chunks):
        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"""
        You are a highly precise research assistant. 
        Answer the question using ONLY the provided context.
        If the information is missing, state that you don't know.

        CONTEXT:
        {context}

        QUESTION:
        {query}
        """
        
        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text

def main():
    if not API_KEY:
        print("Please set GEMINI_API_KEY in your .env file.")
        return

    rag = SemanticRAG(API_KEY)
    
    # 1. Prepare Data
    if rag.load_and_chunk(RAG_RESOURCE_PATH):
        
        while True:
            user_query = input("\nAsk a question (or 'exit'): ").strip()
            if user_query.lower() == 'exit': break
            
            # 2. Retrieve
            print("Searching knowledge base...")
            relevant_chunks = rag.semantic_search(user_query)
            
            # 3. Generate
            print("Synthesizing answer...")
            answer = rag.generate_answer(user_query, relevant_chunks)
            
            print(f"\nAI RESPONSE:\n{answer}")

if __name__ == "__main__":
    main()
