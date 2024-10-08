from huggingface_hub import login
login("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")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import pipeline


loader = PyPDFLoader("Neonatal.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(documents)


hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(documents, embedding_function)

query = "A 3-month-old boy presents to his pediatrician with persistent diarrhea, oral candidiasis, and signs and symptoms suggestive of respiratory syncytial virus (RSV) pneumonia. His weight is in the 10th percentile. He is being evaluated for an immunodeficiency disease. Laboratory results for the HIV are negative by PCR. Which of the following is the most likely cause of these findings in this patient?"
retrieved_results = db.similarity_search(query)

if retrieved_results:
    print("Retrieved Content: ", retrieved_results[0].page_content)
else:
    print("No relevant information found.")


generator = pipeline("text-generation", model="NousResearch/Meta-Llama-3.1-8B")

if retrieved_results:
    retrieved_content = retrieved_results[0].page_content

    response = generator(f"Based on the content: {retrieved_content}\nQuestion: {query}\nAnswer:", max_new_tokens=50)
    print("Generated Response: ", response[0]['generated_text'])
else:
    print("No relevant content to generate an answer.")
