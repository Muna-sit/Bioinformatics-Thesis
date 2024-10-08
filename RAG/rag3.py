import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os

loader = PyPDFLoader("Neonatal.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(documents)


embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


db = Chroma.from_documents(documents, embedding_function)


query = "A 3-month-old boy presents to his pediatrician with persistent diarrhea, oral candidiasis, and signs and symptoms suggestive of respiratory syncytial virus (RSV) pneumonia. His weight is in the 10th percentile. He is being evaluated for an immunodeficiency disease. Laboratory results for HIV are negative by PCR. Which of the following is the most likely cause of these findings in this patient?"


retrieved_results = db.similarity_search(query)

if retrieved_results:
    retrieved_content = retrieved_results[0].page_content
    print("Retrieved Content: ", retrieved_content)


    prompt = f"Context: {retrieved_content}\nQuestion: {query}\nAnswer: "
    

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )

    answer = response.get('message', {}).get('content', '').strip()
    print("Generated Response: ", answer)
else:
    print("No relevant information found.")
