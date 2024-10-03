import ollama
import json
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Load the BERT tokenizer and model
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
    prompt += "Answer with the letter only (A, B, C, or D). Be sure to carefully consider all options"
    
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

# Load the dataset
data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

total = len(data)
correct = 0
total_similarity = 0.0

print(f"Total Questions to Process: {total}")

# Process each entry in the dataset
for idx, entry in enumerate(data):
    question = entry['question']
    options = entry['options']
    correct_answer_key = entry['answer_idx']

    predicted_answer = llama_predict(question, options)

    correct_answer_text = options[correct_answer_key]
    predicted_answer_text = options.get(predicted_answer, "")  # Use .get() to avoid KeyError

    similarity = compute_similarity(predicted_answer_text, correct_answer_text)
    total_similarity += similarity  # Accumulate similarity scores

    print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answer_key}, Similarity: {similarity:.4f}")

    if predicted_answer == correct_answer_key:
        correct += 1

    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{total} questions...")

# Calculate accuracy and average similarity
accuracy = (correct / total) * 100 if total > 0 else 0
average_similarity = total_similarity / total if total > 0 else 0  # Calculate average similarity

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average BERT Similarity Score: {average_similarity:.4f}")

