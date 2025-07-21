import json
import sys # For stderr
from typing import TYPE_CHECKING, Dict, Any, Optional

from pydantic import ValidationError
from pydantic_models import SQLGenerationResponse
from error_handler_module import handle_exception

# Import for RAG
import vector_store_module
import token_utils

if TYPE_CHECKING:
    # postgres_copilot_chat now defines LiteLLMMcpClient
    from postgres_copilot_chat import LiteLLMMcpClient

MAX_SQL_GEN_RETRIES = 4 # Number of retries for the LLM to fix its own JSON output or SQL errors (Total 5 attempts)
MAX_SQL_EXECUTION_RETRIES = 5 # Number of retries for the LLM to fix SQL execution errors

async def generate_sql_query(
    client: 'LiteLLMMcpClient', 
    natural_language_question: str,
    schema_and_sample_data: Optional[Dict[str, Any]], # Combined DDL and sample data
    insights_markdown_content: Optional[str], # Content of summarized_insights.md
    row_limit_for_preview: int = 1 # Added for controlling preview rows
) -> Dict[str, Any]:
    """
    Generates an SQL query based on a natural language question, schema data, and insights using LiteLLM.
    Validates the LLM's response and executes the query for verification.

    Args:
        client: The LiteLLMMcpClient instance.
        natural_language_question: The user's question.
        schema_and_sample_data: Dictionary containing schema (DDL-like) and sample data for tables.
        insights_markdown_content: String content of the cumulative insights markdown file.

    Returns:
        A dictionary containing the SQL query, explanation, execution results, and user message.
    """
    # For LiteLLM, client.session is the primary check for MCP connection.
    # The LLM interaction is handled by methods within the client.
    if not client.session or not client.current_db_name_identifier: # Added check for db_name_identifier
        return {
            "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
            "message_to_user": "Error: MCP session or DB identifier not available for SQL generation."
        }

    # --- RAG: Retrieve Few-Shot Examples ---
    few_shot_examples_str = "No similar approved queries found to use as examples."
    # Attempt to retrieve RAG examples, but proceed even if it fails.
    try:
        # Using the hardcoded threshold from vector_store_module directly
        current_rag_threshold = vector_store_module.LITELLM_RAG_THRESHOLD

        similar_pairs = vector_store_module.search_similar_nlqs(
            db_name_identifier=client.current_db_name_identifier,
            query_nlq=natural_language_question,
            k=3, # Retrieve top 3 for few-shot prompting
            threshold=current_rag_threshold
        )
        if similar_pairs:
            examples_parts = ["Here are some examples of approved natural language questions and their corresponding SQL queries for this database:\n"]
            for i, pair in enumerate(similar_pairs):
                examples_parts.append(f"Example {i+1}:")
                examples_parts.append(f"  Natural Language Question: \"{pair['nlq']}\"")
                examples_parts.append(f"  SQL Query: ```sql\n{pair['sql']}\n```")
            few_shot_examples_str = "\n".join(examples_parts)
            pass
        else:
            pass
            
    except Exception as e_rag:
        handle_exception(e_rag, natural_language_question, {"context": "RAG few-shot example retrieval"})
    # --- End RAG ---

    # Prepare context strings for the prompt
    schema_context_str = "No schema or sample data provided."
    if schema_and_sample_data:
        try:
            schema_context_str = json.dumps(schema_and_sample_data, indent=2)
        except TypeError as e:
            schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
            handle_exception(e, natural_language_question, {"context": "Serializing schema data"})


    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content

    # Initial prompt for SQL generation
    current_prompt = (
        f"You are an expert PostgreSQL SQL writer. Based on the following database schema, sample data, "
        f"cumulative insights, and the natural language question, generate an appropriate SQL query "
        f"(must start with SELECT) and a brief explanation.\n\n"
        f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
        f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
        f"FEW-SHOT EXAMPLES (Use these to guide your SQL generation if relevant):\n{few_shot_examples_str}\n\n"
        f"NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
        f"Respond ONLY with a single JSON object matching this structure: "
        f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
        f"Ensure the SQL query strictly starts with 'SELECT'."
    )

    llm_response_text = "" # Will store the final text part of LLM response
    parsed_response: Optional[SQLGenerationResponse] = None
    last_error_feedback_to_llm = "" # This will be a user-role message for LiteLLM

    # Initial messages list for LiteLLM
    # System prompt is already part of client.conversation_history if provided
    messages_for_llm = client.conversation_history[:] 

    for attempt in range(MAX_SQL_GEN_RETRIES + 1): 
        # print(f"Attempting SQL generation (Attempt {attempt + 1}/{MAX_SQL_GEN_RETRIES + 1})...") # Internal Detail
        
        user_message_content = ""
        if attempt > 0 and last_error_feedback_to_llm: # This implies a retry
            user_message_content = (
                f"{last_error_feedback_to_llm}\n\n" # last_error_feedback_to_llm is the *previous* error message from assistant
                f"Based on the previous error, please try again.\n"
                f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                f"CUMULATIVE INSIGHTS:\n```markdown\n{insights_context_str}\n```\n\n"
                f"FEW-SHOT EXAMPLES:\n{few_shot_examples_str}\n\n" # Added few-shot examples to retry prompt
                f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                f"Respond ONLY with a single JSON object matching the structure: "
                f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                f"Ensure the SQL query strictly starts with 'SELECT'."
            )
        else: # First attempt
            user_message_content = current_prompt # current_prompt is the initial full prompt

        # Add the current user prompt to messages for this specific call
        # We use a temporary list for the call to avoid polluting main history with retries until success
        current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]

        try:
            # We expect a JSON response, so no tools are passed for this specific call.
            schema_tokens = token_utils.count_tokens(schema_context_str, client.model_name, client.provider)
            llm_response_obj = await client._send_message_to_llm(current_call_messages, natural_language_question, schema_tokens)
            # _send_message_to_llm adds the user prompt to client.conversation_history
            # _process_llm_response will add the assistant's response
            
            # _process_llm_response returns (text_content, tool_calls_made)
            # For this call, we expect tool_calls_made to be False.
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_feedback_to_llm = "Your response included an unexpected tool call. Please provide the JSON response directly."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM attempted tool call instead of providing JSON for SQL generation.")

            # print(f"LLM processed response for SQL gen (Attempt {attempt + 1}): {llm_response_text}") # Internal Detail

            cleaned_llm_response_text = llm_response_text.strip()
            if cleaned_llm_response_text.startswith("```json"):
                cleaned_llm_response_text = cleaned_llm_response_text[7:]
            if cleaned_llm_response_text.endswith("```"):
                cleaned_llm_response_text = cleaned_llm_response_text[:-3]
            cleaned_llm_response_text = cleaned_llm_response_text.strip()

            if not cleaned_llm_response_text: # Handle empty response after stripping
                last_error_feedback_to_llm = "Your response was empty. Please provide the JSON output."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM provided an empty response for SQL generation.")

            parsed_response = SQLGenerationResponse.model_validate_json(cleaned_llm_response_text)
            
            if not parsed_response.sql_query:
                error_detail = parsed_response.error_message or "LLM did not provide an SQL query in the 'sql_query' field."
                last_error_feedback_to_llm = (
                    f"Your previous attempt did not produce an SQL query in the 'sql_query' field. LLM message: '{error_detail}'. "
                    f"Ensure 'sql_query' field is populated with a valid SQL string."
                )
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception(f"LLM failed to provide SQL query after retries. Last message: {error_detail}")

            if not parsed_response.sql_query.strip().upper().startswith("SELECT"):
                last_error_feedback_to_llm = (
                    f"Your generated SQL query was: ```sql\n{parsed_response.sql_query}\n```\n"
                    f"This query does not start with SELECT, which is a requirement. Please regenerate a valid SELECT query."
                )
                if attempt < MAX_SQL_GEN_RETRIES:
                    parsed_response = None # Invalidate this response
                    continue
                else: raise Exception("LLM failed to generate a SELECT query after retries.")
            
            # print("SQL Generation successful and format validated.") # Internal Detail
            break 

        except Exception as e:
            context = {
                "context": "SQL generation loop", 
                "attempt": attempt + 1, 
                "llm_response_text": llm_response_text,
                "db_name_identifier": client.current_db_name_identifier
            }
            user_message = handle_exception(e, natural_language_question, context)
            last_error_feedback_to_llm = f"A validation error occurred: {user_message}. Please try again."
            if attempt == MAX_SQL_GEN_RETRIES:
                # The message is now displayed by the caller in postgres_copilot_chat.py
                # We just pass it along in the dictionary.
                return {
                    "sql_query": None, "explanation": None, "execution_result": None, "execution_error": user_message,
                    "message_to_user": "Failed to generate SQL query after all attempts." # Generic message for the main part
                }
    # End of retry loop

    if not parsed_response or not parsed_response.sql_query:
        return {
            "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
            "message_to_user": "Failed to generate SQL query after all attempts."
        }

    sql_to_execute = parsed_response.sql_query
    explanation = parsed_response.explanation
    execution_result = None
    execution_error = None
    
    # Execute SQL with retry logic for execution errors
    original_sql = sql_to_execute
    original_explanation = explanation
    
    for exec_attempt in range(MAX_SQL_EXECUTION_RETRIES + 1):  
        # print(f"Executing SQL (Attempt {exec_attempt + 1}/{MAX_SQL_EXECUTION_RETRIES + 1}): {sql_to_execute}") # Internal
        
        try:
            exec_result_obj = await client.session.call_tool(
                "execute_postgres_query",
                {"query": sql_to_execute, "row_limit": row_limit_for_preview}
            )
            raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj, "execute_postgres_query")

            try:
                # Attempt to parse the output if it's a JSON string
                if isinstance(raw_exec_output, str):
                    exec_data = json.loads(raw_exec_output)
                else:
                    exec_data = raw_exec_output

                # Check for error in the structured response
                if isinstance(exec_data, dict) and exec_data.get("status") == "error":
                    execution_error = exec_data.get("message", "Unknown execution error.")
                    
                    if exec_attempt < MAX_SQL_EXECUTION_RETRIES:
                        fix_user_message_content = (
                            f"The previously generated SQL query resulted in an execution error.\n"
                            f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                            f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                            f"SQL QUERY WITH ERROR:\n```sql\n{sql_to_execute}\n```\n\n"
                            f"EXECUTION ERROR MESSAGE:\n{execution_error}\n\n"
                            f"Please provide a corrected SQL query. For the explanation, describe how the *corrected* SQL query answers the original NATURAL LANGUAGE QUESTION. Do not mention the error, the incorrect query, or the process of fixing it in your explanation. Focus solely on explaining the logic of the corrected query in relation to the user's question.\n"
                            f"Respond ONLY with a single JSON object matching this structure: "
                            f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                            f"Ensure the SQL query strictly starts with 'SELECT'."
                        )
                        
                        fix_call_messages = client.conversation_history + [{"role": "user", "content": fix_user_message_content}]
                        
                        try:
                            schema_tokens = token_utils.count_tokens(schema_context_str, client.model_name, client.provider)
                            fix_llm_response_obj = await client._send_message_to_llm(fix_call_messages, natural_language_question, schema_tokens)
                            fix_text, fix_tool_calls_made = await client._process_llm_response(fix_llm_response_obj)

                            if fix_tool_calls_made:
                                continue

                            if fix_text.startswith("```json"): fix_text = fix_text[7:]
                            if fix_text.endswith("```"): fix_text = fix_text[:-3]
                            fix_text = fix_text.strip()
                            
                            if not fix_text:
                                continue

                            fixed_response = SQLGenerationResponse.model_validate_json(fix_text)
                            
                            if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                sql_to_execute = fixed_response.sql_query
                                explanation = fixed_response.explanation
                                execution_error = None 
                                continue 
                        except Exception as fix_e:
                            handle_exception(fix_e, natural_language_question, {"context": "LLM SQL fix attempt", "attempt": exec_attempt + 1})
                else:
                    execution_result = exec_data
                    execution_error = None
                    break
            except json.JSONDecodeError:
                # If output is not a valid JSON, treat it as a potential success or non-error message
                execution_result = raw_exec_output
                execution_error = None
                break
                
        except Exception as e:
            execution_error = handle_exception(e, natural_language_question, {"context": "SQL execution loop", "attempt": exec_attempt + 1, "sql_to_execute": sql_to_execute})
            if exec_attempt < MAX_SQL_EXECUTION_RETRIES:
                continue
    
    if execution_error and sql_to_execute != original_sql:
        # print("All SQL fix attempts failed. Reverting to original SQL for display.") # Internal detail
        sql_to_execute = original_sql
        explanation = original_explanation

    user_message = f"Generated SQL:\n```sql\n{sql_to_execute}\n```\n\nExplanation:\n{explanation}\n\n"
    
    # The main chat loop will display the execution result or error to the user.
    # This module just returns the necessary data.
    # The print statements for execution success/error are handled in postgres_copilot_chat.py's feedback loop.

    # The user message will be augmented in postgres_copilot_chat.py to include display_few_shot_examples
    user_message += "If this is correct, use '/approved'. If not, use '/feedback Your feedback text'."
    
    # Return the generated SQL and explanation.
    # The few-shot examples used for RAG are not directly returned here,
    # as the main chat loop will fetch them again for display using the display_threshold.
    return {
        "sql_query": sql_to_execute,
        "explanation": explanation,
        "execution_result": execution_result,
        "execution_error": execution_error,
        "message_to_user": user_message
        # "rag_examples_used": similar_pairs # Optionally return this if needed for debugging or direct display
    }
