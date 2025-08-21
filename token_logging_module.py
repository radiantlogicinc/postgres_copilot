import json
import os
from datetime import datetime
from typing import Dict, Any

import memory_module

def log_token_usage(
    origin_script: str,
    origin_line: int,
    user_query: str,
    prompt: str,
    model_id: str,
    prompt_tokens: int = 0,
    schema_tokens: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    llm_response: str = "",
    command: str = None,
    insights_tokens: int = 0,
    other_prompt_tokens: int = 0,
    hyde_tokens: int = 0,
    sql_generation_tokens: int = 0,
    source: str = None
):
    """
    Logs the token usage of an LLM call to a JSON file.

    Args:
        origin_script (str): The name of the script where the LLM call originated.
        origin_line (int): The line number of the LLM call.
        user_query (str): The user's natural language query.
        prompt (str): The full prompt sent to the LLM.
        model_id (str): The ID of the model used for the LLM call.
        prompt_tokens (int): The number of tokens in the prompt.
        schema_tokens (int): The number of tokens used for schema information.
        input_tokens (int): The total number of input tokens reported by the API.
        output_tokens (int): The number of output tokens.
        llm_response (str): The response from the LLM.
        command (str, optional): The command context (/generate_sql, /revise, /feedback).
        insights_tokens (int): The number of tokens used for insights.
        other_prompt_tokens (int): The number of tokens used for other prompt components.
        hyde_tokens (int): The number of tokens used in the HyDE process.
        sql_generation_tokens (int): The number of tokens used in SQL generation.
        source (str, optional): The source of the token usage (e.g., 'hyde', 'sql_generation').
    """
    log_dir = os.path.join(memory_module.get_memory_base_path(), 'logs')
    log_file = os.path.join(log_dir, 'token_usage.json')

    # Create token details dictionary
    token_details = {
        'prompt_tokens': prompt_tokens,
        'schema_tokens': schema_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'insights_tokens': insights_tokens,
        'other_prompt_tokens': other_prompt_tokens,
        'hyde_tokens': hyde_tokens,
        'sql_generation_tokens': sql_generation_tokens,
        'total_tokens': input_tokens + output_tokens,
        'source': source
    }
    
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'origin_script': os.path.basename(origin_script),
        'origin_line': origin_line,
        'user_query': user_query,
        'prompt': prompt,
        'model_id': model_id,
        'command': command,
        'token_details': token_details,
        'llm_response_preview': llm_response[:200] if llm_response else ""
    }

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
