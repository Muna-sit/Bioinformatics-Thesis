import ollama
import json

def llama_predict(question, options):
   
    prompt = f"Q: {question}\n"
    for key, value in options.items():
        prompt += f"{key}: {value}\n"
    prompt += "Answer with the letter only (A, B, C, or D)."

    
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

   
    print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answer_key}")

    
    if predicted_answer == correct_answer_key:
        correct += 1

   
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{total} questions...")


accuracy = (correct / total) * 100
print(f"Accuracy: {accuracy:.2f}%")
