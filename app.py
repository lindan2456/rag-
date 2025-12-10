# app.py

from fastapi import FastAPI
from rag_pipe import load_text_from_s3, add_document_to_db, answer_query

app = FastAPI()

# Replace these with your actual AWS values
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET = "my-rag-bucket-vishnu"
KEY = "s3.txt.rtf"  # example: docs/mytext.txt


# Load text from S3 and index it when the app starts
@app.on_event("startup")
def load_data():
    text = load_text_from_s3(BUCKET, KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY)
    add_document_to_db(text)
    print("Text loaded from S3 and indexed in Chroma!")


# Chat endpoint
@app.get("/chat")
def chat(q: str):
    answer = answer_query(q)
    return {"answer": answer}

