import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import json


loader = PyPDFLoader("Neonatal.pdf")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
documents = text_splitter.split_documents(documents)


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding_function)


file_path = 'EnglishTest.jsonl'
with open(file_path, 'r') as f:
    data = [json.loads(line) for line in f]


def get_model_answer(question, options):
    prompt = f"Q: {question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "Answer with the letter only (A, B, C, or D). Be sure to carefully consider all options."

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )

    return response.get('message', {}).get('content', '').strip()


results = []


score_threshold = 0.5 
for entry in data:
    question = entry['question']
    options = entry['options']
    correct_answer_key = entry['answer_idx']

  
    extended_query = f"{question}\nOptions: " + ", ".join([f"{key}: {value}" for key, value in options.items()])

 
    retrieved_results = db.similarity_search(extended_query)

   
    if retrieved_results and retrieved_results[0].score >= score_threshold:
        retrieved_content = retrieved_results[0].page_content  
        results.append((retrieved_content, correct_answer_key))  
    else:
       
        answer = get_model_answer(question, options)
        results.append(("No relevant information found.", answer))  


with open('output.txt', 'w') as f:
    for retrieved_content, answer in results:
        f.write(f"Retrieved Content: {retrieved_content}\n")
        f.write(f"Predicted Answer: {answer}\n\n")
