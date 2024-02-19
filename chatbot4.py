import openai

# Set your OpenAI API key
openai.api_key = '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'

def healthcare_chatbot(prompt):
    # Define the prompt for the healthcare chatbot
    prompt = f"The following is a conversation with a healthcare chatbot. The bot is designed to provide information and assistance related to health. \n\nUser: {prompt}\nBot:"

    # Generate a response from the chatbot
    response = openai.Completion.create(
        engine="text-davinci-003",  # You can experiment with different engines
        prompt=prompt,
        temperature=0.7,  # Adjust temperature for response creativity
        max_tokens=150  # Adjust max tokens for response length
    )

    # Extract the generated text from the response
    bot_response = response.choices[0].text.strip()

    return bot_response
