# rag_functions.py

import boto3
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline


# ----------------------------
# 1. LOAD TEXT FROM S3
# ----------------------------
def load_text_from_s3(bucket, key, aws_access_key, aws_secret_key):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8")
    return text


# ----------------------------
# 2. INITIALIZE EMBEDDINGS + CHROMA
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()

# Create collection to store text chunks
collection = client.create_collection(name="docs")


# ----------------------------
# 3. ADD DOCUMENT TO VECTOR DB
# ----------------------------
def add_document_to_db(text):
    chunks = text.split(".")   # very simple splitting into sentences
    embeddings = embed_model.encode(chunks)

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings
    )


# ----------------------------
# 4. RETRIEVE MATCHING DOCUMENTS
# ----------------------------
def retrieve(query):
    query_emb = embed_model.encode([query])[0]
    result = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )
    return result["documents"][0]


# ----------------------------
# 5. SIMPLE LLM GENERATION
# ----------------------------
generator = pipeline("text-generation", model="gpt2")

def answer_query(query):
    docs = retrieve(query)
    context = "\n".join(docs)

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=150, num_return_sequences=1)
    return response[0]["generated_text"]
