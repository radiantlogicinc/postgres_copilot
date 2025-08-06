import json
import os
from datetime import datetime

import memory_module

def log_token_usage(origin_script, origin_line, user_query, prompt, prompt_tokens, schema_tokens, input_tokens, output_tokens, llm_response, model_id, command=None):
    """
    Logs the token usage of an LLM call to a JSON file.

    Args:
        origin_script (str): The name of the script where the LLM call originated.
        origin_line (int): The line number of the LLM call.
        user_query (str): The user's natural language query.
        prompt (str): The full prompt sent to the LLM.
        prompt_tokens (int): The number of tokens in the prompt.
        schema_tokens (int): The number of tokens in the schema.
        input_tokens (int): The total number of input tokens.
        output_tokens (int): The number of output tokens from the LLM response.
        llm_response (str): The response from the LLM.
        model_id (str): The ID of the model used for the LLM call.
        command (str, optional): The command context (/generate_sql, /revise, /feedback).
    """
    log_dir = os.path.join(memory_module.get_memory_base_path(), 'logs')
    log_file = os.path.join(log_dir, 'token_usage.json')

    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'origin_script': os.path.basename(origin_script),
        'origin_line': origin_line,
        'user_query': user_query,
        'prompt': prompt,
        'prompt_tokens': prompt_tokens,
        'schema_tokens': schema_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'llm_response': llm_response,
        'model_id': model_id,
        'command': command,
    }

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
