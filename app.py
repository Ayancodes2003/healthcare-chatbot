from flask import Flask, render_template, request

# Import the healthcare_chatbot function from chat_bot2.py
from chat_bot2 import healthcare_chatbot

app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle chatbot functionality
@app.route('/get', methods=['GET'])
def get_bot_response():
    user_input = request.args.get('msg')
    # Call the healthcare_chatbot function with the user input
    # and get the response
    response = healthcare_chatbot(user_input)
    # Return the response as a string
    return str(response)

if __name__ == '__main__':
    app.run(debug=True)
