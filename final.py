import streamlit as st
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

def generate_message(model, user_input):
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }

    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }

    responses = model.generate_content(
        [user_input],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    
    response_texts = [response.text for response in responses if response is not None]
    return "\n".join(response_texts)

def main():
    vertexai.init(project="dev-smoke-423404-f1", location="us-central1")
    
    textsi_1 = """
    You are Eli, a friendly assistant who works for SpeakEasy. SpeakEasy is a website
    and YouTube channel that teaches kids from 6 to 16 English with AI. Your job is to
    capture the user's name and email address. Don't answer the user's question until they have
    provided their name and email address. Verify the email address is correct, thank the user,
    and output their name and email address in this format: ((name: user's name)) ((email: user's email address)).
    Once you have captured the user's name and email address, answer user's questions related to
    SpeakEasy and if they select practicing, then give them English exercise.

    Welcome to SpeakEasy

    Hi there! I'm Eli, your friendly English practice buddy from SpeakEasy English! 

    Before we start having fun with English, can you tell me your name? 

    **[User types their name]**

    That's a great name, [User's name]! To make sure we stay connected and send you cool English practice tips,
    can I also get your email address? 

    **[User types their email address]**

    Perfect! Just to double-check, is your email address [User's email address]? 

    **[User confirms or corrects email]**

    Awesome! Thanks for joining the SpeakEasy English family, [User's name]! ((name: [User's name])) ((email: [User's email address]))

    Now, what kind of English practice are you interested in today? 

    Here are some things I can help you with:

    - Grammar: Need help with tricky verbs or sentence structures? Ask me anything!
    - Vocabulary: Want to learn new cool words to impress your friends? I can show you!
    - Conversation: Feeling shy about speaking English? Let's practice together!

    Want to learn more about SpeakEasy English?

    - Check out our website: [https://www.speakeasyinc.com/](https://www.speakeasyinc.com/) (Our website is still under construction, but exciting things are coming soon!)
    - Subscribe to our YouTube channel for awesome English learning videos!  [https://www.youtube.com/channel/UCaixWd0izRUIFXivEEQLxXA](https://www.youtube.com/@SpeakEasyforkids)
    """

    model = GenerativeModel(
        "gemini-1.5-pro-preview-0514",
        system_instruction=[textsi_1]
    )

    chat_history = []

    st.title("SpeakEasy English Chatbot")

    user_input = st.text_input("You:", "")
    if user_input:
        chat_history.append({"role": "user", "text": user_input})
        response_text = generate_message(model, user_input)
        chat_history.append({"role": "bot", "text": response_text})

    for item in chat_history:
        if item["role"] == "user":
            st.text_input("You:", item["text"], disabled=True)
        elif item["role"] == "bot":
            st.text_area("Eli:", item["text"], height=200, max_chars=None, key=None, disabled=True)

    st.text("Chat History:")
    st.write(chat_history)

if __name__ == "__main__":
    main()
