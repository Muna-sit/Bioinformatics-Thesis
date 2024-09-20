import ollama
import json
import concurrent.futures

def llama_predict(question, options):
    prompt = f"Q: {question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "Answer with only one of the letters (A, B, C, or D). Be sure to carefully consider all options.."

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}]
    )
    
    answer = response.get('message', {}).get('content', '').strip()

    # Match the model's prediction with the available options
    for key in options:
        if key in answer:  # Match by option label (A, B, C, D)
            return key
    return None

def process_question(entry):
    question = entry['question']
    options = entry['options']
    correct_answer_key = entry['answer_idx']

    # Get the predicted answer from LLaMA 3.1 via Ollama
    predicted_answer = llama_predict(question, options)

    # Print predicted and correct answer
    print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answer_key}")

    # Return whether the prediction is correct or not
    return predicted_answer == correct_answer_key

# Load the dataset
file_path = 'EnglishTest.jsonl'

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Process only 100 questions
data = data[:100]

print(f"Total Questions to Process: {len(data)}")

correct = 0

# Use ThreadPoolExecutor for parallel execution
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_question, data))

# Calculate total correct answers
correct = sum(results)

# Calculate accuracy
accuracy = (correct / len(data)) * 100
print(f"Accuracy: {accuracy:.2f}%")
