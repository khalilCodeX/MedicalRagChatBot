import gradio as gr
import genai_llm as genai
from langchain_core.messages import HumanMessage, AIMessage

# Initialize bot
bot = genai.genai_llm()
bot.set_embedding_vector()

def chat_function(message, history):
    """Process user message and return response with chat history context"""
    
    # Convert Gradio history format to LangChain message format
    # Gradio history: [["user msg", "bot msg"], ...]
    # LangChain expects: [HumanMessage, AIMessage, ...]
    bot.chat_history = []
    if history:
        for user_msg, bot_msg in history:
            bot.chat_history.append(HumanMessage(content=user_msg))
            bot.chat_history.append(AIMessage(content=bot_msg))
    
    # Get response (this will add current message to history internally)
    response = bot.create_chain(message)
    return response

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_function,
    title="Medical AI Assistant",
    description="Ask me any medical questions based on our database",
    examples=[
        "What are the symptoms of diabetes?",
        "How to treat a headache?",
        "What causes high blood pressure?"
    ],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True gives you a public URL