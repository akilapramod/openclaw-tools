import functools
import logging

# Global counters for the session
SESSION_TOTAL_IN = 0
SESSION_TOTAL_OUT = 0

def token_usage_hook(func):
    """Decorator to intercept LLM calls and log exact token metrics."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global SESSION_TOTAL_IN, SESSION_TOTAL_OUT
        
        # 1. Execute the actual LLM API call
        response = func(*args, **kwargs)
        
        try:
            tokens_in = 0
            tokens_out = 0
            
            # 2. Extract usage (Handles both Gemini and OpenAI SDK formats)
            if hasattr(response, 'usage_metadata'):
                # Google Generative AI (Gemini) format
                tokens_in = response.usage_metadata.prompt_token_count
                tokens_out = response.usage_metadata.candidates_token_count
            elif hasattr(response, 'usage'):
                # OpenAI / LangChain format
                tokens_in = response.usage.prompt_tokens
                tokens_out = response.usage.completion_tokens
            elif isinstance(response, dict) and 'usage' in response:
                # Raw dictionary format
                tokens_in = response['usage'].get('prompt_tokens', 0)
                tokens_out = response['usage'].get('completion_tokens', 0)
            else:
                return response
                
            # 3. Update Totals
            SESSION_TOTAL_IN += tokens_in
            SESSION_TOTAL_OUT += tokens_out
            
            # 4. Print the exact requested FinOps format
            print(f"in: {tokens_in} out: {tokens_out} total_in: {SESSION_TOTAL_IN} total_out: {SESSION_TOTAL_OUT}")
            
        except Exception as e:
            logging.debug(f"Token hook parsing skipped: {e}")
            
        return response
    return wrapper
