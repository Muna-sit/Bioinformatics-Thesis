import ollama
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embedding(text):
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
  
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def compute_similarity(pred_answer, correct_answer):
    
    pred_embedding = get_bert_embedding(pred_answer)
    correct_embedding = get_bert_embedding(correct_answer)
    
    
    similarity = cosine_similarity(pred_embedding.unsqueeze(0), correct_embedding.unsqueeze(0))
    return similarity.item()

def llama_predict(question, options):
   
    prompt = f"Q: {question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "Answer with the letter only (A, B, C, or D).Be sure to carefully consider all options"

   
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )

   
    answer = response.get('message', {}).get('content', '').strip()

   
    for key in options:
        if key in answer: 
            return key
    return None


file_path = 'EnglishTest.jsonl'

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))


data = data[:100]

correct = 0
total = len(data)

print(f"Total Questions to Process: {total}")

for idx, entry in enumerate(data):
    question = entry['question']
    options = entry['options']
    correct_answer_key = entry['answer_idx']  

   
    predicted_answer = llama_predict(question, options)

   
    correct_answer_text = options[correct_answer_key]
    predicted_answer_text = options[predicted_answer] if predicted_answer else ""


    similarity = compute_similarity(predicted_answer_text, correct_answer_text)


    print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answer_key}, Similarity: {similarity:.4f}")


    if predicted_answer == correct_answer_key:
        correct += 1


    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{total} questions...")


accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")


    """import ollama
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

# Function to compute cosine similarity between predicted and correct answer
def compute_similarity(pred_answer, correct_answer):
    pred_embedding = get_bert_embedding(pred_answer)
    correct_embedding = get_bert_embedding(correct_answer)
    similarity = cosine_similarity(pred_embedding.unsqueeze(0), correct_embedding.unsqueeze(0))
    return similarity.item()

# Function to interact with the LLaMA model via Ollama
def llama_predict(question, options):
    prompt = f"Q: {question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "Answer with the letter only (A, B, C, or D). Be sure to carefully consider all options"

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )

    answer = response.get('message', {}).get('content', '').strip()

    for key in options:
        if key in answer:  # Match by option label (A, B, C, D)
            return key
    return None

# Load the dataset
file_path = 'EnglishTest.jsonl'

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Process the entire dataset
correct = 0
total = len(data)
total_similarity = 0.0

print(f"Total Questions to Process: {total}")

# Loop over each entry in the dataset
for idx, entry in enumerate(data):
    question = entry['question']
    options = entry['options']
    correct_answer_key = list(options.keys())[entry['answer_idx']]  # Convert index to the actual option key (A, B, C, or D)

    # Get the predicted answer from LLaMA 3.1 via Ollama
    predicted_answer = llama_predict(question, options)

    # Get the correct and predicted answer text
    correct_answer_text = options[correct_answer_key]
    predicted_answer_text = options.get(predicted_answer, "")

    # Compute BERT similarity
    similarity = compute_similarity(predicted_answer_text, correct_answer_text) if predicted_answer else 0.0
    total_similarity += similarity

    # Print the predicted and correct answers with the similarity score
    print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answer_key}, Similarity: {similarity:.4f}")

    # Check if the prediction matches the actual correct answer
    if predicted_answer == correct_answer_key:
        correct += 1

    # Print progress every 10 questions
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{total} questions...")

# Calculate accuracy
accuracy = (correct / total) * 100


# Calculate average BERT similarity score
average_similarity = total_similarity / total

# Display the final accuracy and average BERT similarity result
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average BERT Similarity Score: {average_similarity:.4f}")

    """
    