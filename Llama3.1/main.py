import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from huggingface_hub import login
import torch  


login(token="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")


if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

def load_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl('English.jsonl')
print(train_data[:2])  


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


print(f"Model is running on: {device}")


def preprocess_data(examples):
    combined_input = []
    labels = []
   
    for example in examples:
  
        question_with_options = example['question'] + '\n'
        for i, option in enumerate(example['options']):
            question_with_options += f"{i+1}. {option}\n"
       
        combined_input.append(question_with_options)
        labels.append(example['answer'])  
       
    return {
        'input_text': combined_input,
        'labels': labels
    }


train_dataset = Dataset.from_dict({
    'question': [item['question'] for item in train_data],
    'options': [item['options'] for item in train_data],
    'answer': [item['answer'] for item in train_data]
})


train_dataset = train_dataset.map(preprocess_data, batched=True)


def tokenize_data(examples):
    return tokenizer(examples['input_text'], max_length=512, truncation=True, padding='max_length')

train_dataset = train_dataset.map(tokenize_data, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print(f"Trainer is running on: {trainer.args.device}")

trainer.train()


test_data = load_jsonl('EnglishTest.jsonl')

for item in test_data:
    question = item['question']
    options = item['options']
    answer_idx = item['answer_idx'] 
    question_with_options = question + '\n'
    for i, option in enumerate(options):
        question_with_options += f"{i+1}. {option}\n"
   
   
    inputs = tokenizer(question_with_options, return_tensors="pt").to(device)
   

    outputs = model.generate(**inputs)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
   
 
    selected_option = None
    for option in options:
        if option.lower() in predicted_answer.lower():
            selected_option = option
            break
   
    
    if not selected_option:
        selected_option = predicted_answer
   
    print(f"Question: {question}")
    print(f"Predicted Answer: {selected_option}")
    print(f"Options: {options}")
    print(f"Correct Answer (by index): {options[answer_idx]}")
    print("\n")
