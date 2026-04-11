import os
from dotenv import load_dotenv
from google import genai

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load variables from .env next to this script
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
RAG_RESOURCE_PATH = os.getenv("RAG_RESOURCE_PATH", "resources/knowledge.txt")

def resolve_resource_path(file_path):
    if os.path.isabs(file_path):
        return file_path

    return os.path.join(PROJECT_ROOT, file_path)

def check_gemini_key(key):
    if not key:
        print("Error: No API key found. Check your .env file.")
        return None

    try:
        client = genai.Client(api_key=key)
        print("Success! API key is valid.")
        return client
    except Exception as e:
        print("Failed to verify API key.")
        print(f"Error Details: {e}")
        return None

def load_resource_text(file_path):
    resolved_path = resolve_resource_path(file_path)

    if not os.path.exists(resolved_path):
        print(f"Error: Resource file not found at {resolved_path}")
        return None

    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print("Failed to read resource file.")
        print(f"Error Details: {e}")
        return None

def split_text_into_chunks(text, chunk_size=800):
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def simple_keyword_retrieval(chunks, query, top_k=3):
    query_words = set(query.lower().split())
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    top_chunks = [chunk for score, chunk in scored_chunks[:top_k] if score > 0]

    # fallback: if no keyword match, return the first chunk
    if not top_chunks and chunks:
        top_chunks = chunks[:1]

    return top_chunks

def build_rag_prompt(user_query, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)

    return f"""
You are a helpful assistant.
Answer the user's question only using the provided context.
If the answer is not in the context, say clearly: "I could not find that in the provided resource."

Context:
{context}

User Question:
{user_query}
""".strip()

def get_llm_response(client, model_name, prompt):
    if not client:
        print("Error: Gemini client is not available.")
        return

    if not prompt.strip():
        print("Error: Prompt cannot be empty.")
        return

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        print(f"\nResponse:\n{response.text}")
    except Exception as e:
        print("Failed to generate LLM response.")
        print(f"Error Details: {e}")

def run_simple_rag(client, model_name, resource_path, user_query):
    resource_text = load_resource_text(resource_path)
    if not resource_text:
        return

    chunks = split_text_into_chunks(resource_text, chunk_size=800)
    relevant_chunks = simple_keyword_retrieval(chunks, user_query, top_k=3)

    prompt = build_rag_prompt(user_query, relevant_chunks)
    get_llm_response(client, model_name, prompt)

if __name__ == "__main__":
    gemini_client = check_gemini_key(API_KEY)

    if gemini_client:
        user_prompt = input("Enter your question: ").strip()
        run_simple_rag(gemini_client, MODEL_NAME, RAG_RESOURCE_PATH, user_prompt)
