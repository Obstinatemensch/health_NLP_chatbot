import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pickle

# Load the tokenized pairs from the pickle file
with open('tokenized_pairs.pkl', 'rb') as f:
    tokenized_pairs = pickle.load(f)

# Split the tokenized pairs into questions and answers
questions = [pair[0] for pair in tokenized_pairs]
answers = [pair[1] for pair in tokenized_pairs]

# Set up the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Convert the tokenized questions and answers to input features
input_ids = []
attention_masks = []

for question, answer in tqdm.tqdm(zip(questions, answers)):
    encoded_dict = tokenizer.encode_plus(
                        question,
                        answer,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor([1]*len(tokenized_pairs))

# Split the data into training and validation sets
dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Set up the data loaders for training and validation
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 4
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(train_dataloader):
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        
        model.zero_grad()
        
        loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks, labels=batch_labels)
        
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print('Average training loss: {}'.format(avg_train_loss))
    
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    
    for batch in val_dataloader:
        batch_input_ids = batch[0].to(device)
        batch_attention_masks = batch[1].to(device)
        batch_labels = batch[2].to(device)
        
        with torch.no_grad():
            loss, logits = model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks, labels=batch_labels)
        
        total_eval_loss += loss.item()
        
        logits = logits.detach().cpu().numpy()
        label_ids = batch_labels.to('cpu').numpy()
        total_eval_accuracy += (logits.argmax(axis=1) == label_ids).sum()
        
        nb_eval_steps += 1
    
    avg_val_accuracy = total_eval_accuracy / len(val_dataloader.dataset)
    avg_val_loss = total_eval_loss / len(val_dataloader)
    print('Validation accuracy: {}'.format(avg_val_accuracy))
    print('Validation loss: {}'.format(avg_val_loss))

# Test the model
def predict(input_question):
    model.eval()
    encoded_dict = tokenizer.encode_plus(
                        input_question,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_attention_mask = True,
                        return_tensors = 'pt',
                   )
    
    input_ids = encoded_dict['input_ids'].to(device)
    attention_mask = encoded_dict['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)
    
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label = logits.argmax(axis=1)[0]
    
    if label == 1:
        return 'Yes'
    else:
        return 'No'

# Example usage
input_question = 'Are fatigue, lower back pain and throwing up symptoms of pregnancy?'
answer = predict(input_question)
print(answer)