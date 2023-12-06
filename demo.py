import pandas as pd
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import torch.optim as optim
from tqdm import tqdm

# Load the CSV file
df = pd.read_csv('data.csv')

# Split into input (queries) and output (answers)
queries = df['Query'].tolist()
answers = df['Answer'].tolist()

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and encode the inputs
encoded_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')

# Load pre-trained BERT model for fine-tuning
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Define optimizer and learning rate
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# Convert answers to tensors
answers_start_token = tokenizer.encode(answers, add_special_tokens=False, truncation=True, return_tensors='pt')
answers_end_token = tokenizer.encode(answers, add_special_tokens=False, truncation=True, return_tensors='pt')

# Fine-tune the model
num_epochs = 3  # Adjust based on convergence
batch_size = 19  # Adjust based on your resources

model.train()
print("TRAINING...")
for epoch in range(num_epochs):
    num_batches = len(encoded_inputs['input_ids']) // batch_size
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, len(encoded_inputs['input_ids']))

        optimizer.zero_grad()
        batch_inputs = {key: val[start_idx:end_idx] for key, val in encoded_inputs.items()}
        batch_answers_start = answers_start_token[start_idx:end_idx].squeeze(1)  # Remove extra dimension
        batch_answers_end = answers_end_token[start_idx:end_idx].squeeze(1)  # Remove extra dimension

        # Ensure single-target tensors
        batch_answers_start = batch_answers_start.view(-1)
        batch_answers_end = batch_answers_end.view(-1)

        outputs = model(**batch_inputs, start_positions=batch_answers_start, end_positions=batch_answers_end)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Set model to evaluation mode
model.eval()

# Example inference
print("INFERENCE...")
query = "Your query goes here"
encoded_query = tokenizer.encode_plus(query, return_tensors='pt')

# Perform inference
with torch.no_grad():
    output = model(**encoded_query)

# Decode the output to get the answer
answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
answer = tokenizer.decode(encoded_query['input_ids'][0][answer_start:answer_end+1])
