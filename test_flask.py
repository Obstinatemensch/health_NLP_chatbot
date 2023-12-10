from flask import Flask, request, jsonify
from textwrap import fill
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
finetuned_model = None
tokenizer = None
FLAG = False

def init():
    global finetuned_model, tokenizer, FLAG
    last_checkpoint = r"C:\Users\praya\Downloads\NLP\health_NLP_chatbot\results\checkpoint-18000"
    finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(last_checkpoint)
    FLAG = True

def health_bot_response(user_input):
    global finetuned_model, tokenizer, FLAG
    if not FLAG:
        init()
    inputs = tokenizer(user_input, return_tensors="pt")
    outputs = finetuned_model.generate(**inputs, max_length=120)
    answer = tokenizer.decode(outputs[0])
    return fill(answer, width=80)  # Return the response text instead of printing

from flask import Flask, request, jsonify, render_template
from textwrap import fill
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)
finetuned_model = None
tokenizer = None
FLAG = False

# Rest of your code for model initialization and bot response functions...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    bot_response = health_bot_response(user_input)
    return jsonify({'message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
