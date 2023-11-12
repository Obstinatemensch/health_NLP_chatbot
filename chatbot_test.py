import re
import string
import tokenize
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Step 1: Data Preprocessing
def preprocess_data(file_path):
    with open(file_path, 'r') as f:
        data = f.read()
    questions = re.findall('Patient:(.*?)Doctor:', data, re.DOTALL)
    answers = re.findall('Doctor:(.*?)id=', data, re.DOTALL)
    qa_pairs = [{'question': q.strip(), 'answer': a.strip()} for q, a in zip(questions, answers)]
    return qa_pairs

# Step 2: Data Cleaning
def clean_data(qa_pairs):
    cleaned_pairs = []
    for pair in qa_pairs:
        q = pair['question'].lower()
        a = pair['answer'].lower()
        q = q.translate(str.maketrans('', '', string.punctuation))
        a = a.translate(str.maketrans('', '', string.punctuation))
        q_words = nltk.word_tokenize(q)
        a_words = nltk.word_tokenize(a)
        q_words = [w for w in q_words if w not in stopwords.words('english')]
        a_words = [w for w in a_words if w not in stopwords.words('english')]
        pair['question'] = ' '.join(q_words)
        pair['answer'] = ' '.join(a_words)
        cleaned_pairs.append(pair)
    return cleaned_pairs

# Step 3: Tokenization
def tokenize_data(cleaned_pairs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_pairs = []
    for pair in cleaned_pairs:
        q_tokens = tokenizer.encode(pair['question'], add_special_tokens=True)
        a_tokens = tokenizer.encode(pair['answer'], add_special_tokens=True)
        tokenized_pairs.append({'input_ids': q_tokens + a_tokens[1:], 'attention_mask': [1] * len(q_tokens) + [1] * len(a_tokens[1:])})
    return tokenized_pairs

# Step 4: BERT Encoding
def encode_data(tokenized_pairs):
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_pairs = []
    for pair in tokenized_pairs:
        input_ids = torch.tensor(pair['input_ids']).unsqueeze(0).to(device)
        attention_mask = torch.tensor(pair['attention_mask']).unsqueeze(0).to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)
        answer = tokenizer.decode(pair['input_ids'][start_index:end_index+1], skip_special_tokens=True)
        pair['answer'] = answer
        encoded_pairs.append(pair)
    return encoded_pairs
# Step 5: Fine-tuning (not shown)

# Step 6: Testing
def test_model(encoded_pairs, question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_ids = tokenizer.encode(question, add_special_tokens=True)
    attention_mask = [1] * len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = tokenizer.decode(input_ids[0][answer_start:answer_end+1], skip_special_tokens=True)
    return answer

# # Preprocess the data
# qa_pairs = preprocess_data('healthcaremagic_dialogue_4.txt')

# # Clean the data
# cleaned_pairs = clean_data(qa_pairs)

# # Tokenize the data
# tokenized_pairs = tokenize_data(cleaned_pairs)

import pickle
# # Open the file in binary mode
# with open('tokenized_pairs.pkl', 'wb') as f:
#     # Serialize the variable and write it to the file
#     pickle.dump(tokenized_pairs, f)

# exit()

with open('tokenized_pairs.pkl', 'rb') as f:
    # Deserialize the variable and load it back in
    tokenized_pairs = pickle.load(f)
    
print("tokenized_pairs")

# Encode the data using BERT
encoded_pairs = encode_data(tokenized_pairs)

print("encoding complete ****")

# Test the model
question = 'What is the solution for high ESR levels?'
answer = test_model(encoded_pairs, question)
print(answer)
