import tiktoken

def calculate_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encodingf = tiktoken.encoding_for_model(model)
        tokens = encodingf.encode(text)
        return len(tokens)
    except KeyError:
        raise ValueError(f"Model '{model}' not found. Please provide a valid model name.")
    
def calculate_price(token_len, model: str = "gpt-5-nano", price_per_1M_input_tokens: float = 0.05,  price_per_1M_output_tokens: float = 0.40) -> float:

   expected_input_cost = (token_len / 1000000) * price_per_1M_input_tokens
   estimated_ouput_tokens = token_len * 2  # Assuming output tokens are roughly double the input tokens
   expected_output_cost = (estimated_ouput_tokens / 1000000) * price_per_1M_output_tokens

   return expected_input_cost + expected_output_cost
    