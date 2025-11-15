from token_calc import calculate_tokens, calculate_price
import genai_llm as genai


def main():
    bot = genai.genai_llm()
    bot.set_embedding_vector()
    # doc_chunks = bot.tokenize_chunkify_documents()
    # bot.embed_documents(doc_chunks)

    while True:
        user_input = input("Enter your medical query (or type 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Exiting the chat. Goodbye!")
            break
        response = bot.create_chain(user_input)
        print(f"AI Response: {response}")


if __name__ == "__main__":
    main()


