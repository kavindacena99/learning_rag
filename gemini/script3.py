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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001") 
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
                return None
            try:
                reader = PdfReader(resolved_path)
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n".join(pages).strip()
            except Exception as e:
                print(f"Error Details: {e}")
                return None
        return None

    def load_and_chunk(self, file_path):
        text = self.read_resource_text(file_path)
        if not text:
            return False

        size = 800
        self.chunks = [text[i:i+size] for i in range(0, len(text), size)]
        total_chunks = len(self.chunks)
        print(f"Created {total_chunks} chunks. Generating embeddings...")

        batch_size = 100
        self.embeddings = []
        for i in range(0, total_chunks, batch_size):
            batch = self.chunks[i : i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(total_chunks-1)//batch_size + 1}...")
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            self.embeddings.extend([e.values for e in result.embeddings])
        return True

    def semantic_search(self, query, top_k=5):
        query_result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        query_vec = np.array(query_result.embeddings[0].values)

        scores = []
        for i, chunk_vec in enumerate(self.embeddings):
            chunk_vec = np.array(chunk_vec)
            norm_q = np.linalg.norm(query_vec)
            norm_c = np.linalg.norm(chunk_vec)
            score = np.dot(query_vec, chunk_vec) / (norm_q * norm_c)
            # We store the index 'i' so we can reference which chunk it was
            scores.append((score, self.chunks[i], i))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k] # Returns (score, text, index)

    def generate_answer(self, query, context_chunks):
        # We only want the text part of the tuple for the LLM
        context_text = "\n\n---\n\n".join([item[1] for item in context_chunks])
        prompt = f"""
        You are a highly precise research assistant. 
        Answer the question using ONLY the provided context.
        If the information is missing, state that you don't know.

        CONTEXT:
        {context_text}

        QUESTION:
        {query}
        """
        response = self.client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text

def main():
    if not API_KEY:
        print("Please set GEMINI_API_KEY in your .env file.")
        return

    rag = SemanticRAG(API_KEY)
    
    if rag.load_and_chunk(RAG_RESOURCE_PATH):
        while True:
            user_query = input("\nEnter your question (or 'exit'): ").strip()
            if user_query.lower() == 'exit': break
            
            print("Searching knowledge base...")
            # relevant_chunks now contains (score, text, index)
            relevant_data = rag.semantic_search(user_query)
            
            print("Synthesizing answer...")
            answer = rag.generate_answer(user_query, relevant_data)
            
            print(f"\nAI RESPONSE:\n{answer}")
            
            # --- NEW SECTION: SOURCE DISPLAY ---
            print("\n" + "="*30)
            print("SOURCES USED (Top Semantic Matches):")
            for i, (score, text, idx) in enumerate(relevant_data):
                # Clean up text for display (remove newlines, limit length)
                preview = text.replace('\n', ' ')[:150] + "..."
                print(f"[{i+1}] Chunk Index: {idx} (Similarity: {score:.4f})")
                print(f"    Text: {preview}\n")
            print("="*30)

if __name__ == "__main__":
    main()