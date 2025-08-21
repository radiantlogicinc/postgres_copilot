import json
import sys # For stderr
import re
from typing import TYPE_CHECKING, Dict, Any, Optional

from pydantic import ValidationError
from pydantic_models import SQLGenerationResponse
from error_handler_module import handle_exception

# Import for RAG
import vector_store_module
import token_utils
import hyde_module
import hyde_regeneration_module
import memory_module
import join_path_finder

if TYPE_CHECKING:
    # postgres_copilot_chat now defines LiteLLMMcpClient
    from postgres_copilot_chat import LiteLLMMcpClient

MAX_SQL_GEN_RETRIES = 4 # Number of retries for the LLM to fix its own JSON output or SQL errors (Total 5 attempts)
MAX_SQL_EXECUTION_RETRIES = 5 # Number of retries for the LLM to fix SQL execution errors

def _extract_json_from_response(text: str) -> Optional[str]:
    """
    Extracts a JSON object from a string, even if it's embedded in other text.
    Handles markdown code blocks.
    """
    # Regex to find a JSON object within ```json ... ``` or just { ... }
    match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', text, re.DOTALL)
    if match:
        # Prioritize the content of ```json ... ``` if present
        return match.group(1) if match.group(1) else match.group(2)
    return None

def _extract_tables_from_query(sql: str) -> list[str]:
    """
    Extracts table names from a SQL query using a simple regex.
    Looks for tables after FROM and JOIN clauses.
    """
    # This regex finds words that follow 'FROM' or 'JOIN' keywords.
    # It's a simple approach and might need refinement for complex cases (e.g., subqueries, schemas).
    pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)\b'
    tables = re.findall(pattern, sql, re.IGNORECASE)
    return list(set(tables)) # Return unique table names

async def generate_sql_query(
    client: 'LiteLLMMcpClient', 
    natural_language_question: str,
    schema_and_sample_data: Optional[Dict[str, Any]], # Combined DDL and sample data
    insights_markdown_content: Optional[str], # Content of summarized_insights.md
    row_limit_for_preview: int = 1, # Added for controlling preview rows
    command_type: str = "generate_sql" # Added for conversation history management
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

    # --- HyDE: Retrieve Focused Schema Context ---
    hyde_context = ""
    table_names_from_hyde = []
    try:
        hyde_context, table_names_from_hyde = await hyde_module.retrieve_hyde_context(
            nlq=natural_language_question,
            db_name_identifier=client.current_db_name_identifier,
            llm_client=client
        )
    except Exception as e_hyde:
        handle_exception(e_hyde, natural_language_question, {"context": "HyDE Context Retrieval"})
        hyde_context = "Failed to retrieve focused schema context via HyDE."
    # --- End HyDE ---

    # Prepare context strings for the prompt
    schema_context_str = "No schema or sample data provided."
    if schema_and_sample_data:
        try:
            # Full schema is kept as a fallback, but HyDE context is prioritized
            full_schema_str = json.dumps(schema_and_sample_data, indent=2)
            # If HyDE context is available and not an error message, use it. Otherwise, use full schema.
            if hyde_context and "Failed" not in hyde_context and "Error" not in hyde_context:
                schema_context_str = hyde_context
            else:
                schema_context_str = full_schema_str
        except TypeError as e:
            schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
            handle_exception(e, natural_language_question, {"context": "Serializing schema data"})


    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content

    # --- Load Schema Graph and Find Join Path ---
    schema_graph = memory_module.load_schema_graph(client.current_db_name_identifier)
    join_path_str = "No deterministic join path could be constructed."
    if schema_graph and table_names_from_hyde:
        try:
            join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
            if join_clauses:
                join_path_str = "\n".join(join_clauses)
        except Exception as e_join:
            handle_exception(e_join, natural_language_question, {"context": "Join Path Finder"})
            join_path_str = "Error constructing join path."
    # --- End Join Path Finding ---

    # Initial prompt for SQL generation
    current_prompt = (
        f"You are an expert PostgreSQL SQL writer. Based on the following database schema, sample data, "
        f"cumulative insights, and the natural language question, generate an appropriate SQL query "
        f"(must start with SELECT) and a brief explanation.\n\n"
        f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the query):\n```\n{schema_context_str}\n```\n\n"
        f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query):\n```\n{join_path_str}\n```\n\n"
        f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
        f"FEW-SHOT EXAMPLES (Use these to guide your SQL generation if relevant):\n{few_shot_examples_str}\n\n"
        f"NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
        f"IMPORTANT: Your response MUST be a single, valid JSON object. Do not include any other text, explanations, or markdown formatting outside of the JSON structure.\n"
        f"The JSON object must conform to this exact structure: "
        f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}"
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
                f"DETERMINISTIC JOIN PATH:\n```\n{join_path_str}\n```\n\n"
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
            # Count tokens for the complete prompt
            prompt_text = user_message_content
            prompt_tokens = token_utils.count_tokens(current_prompt, client.model_name, client.provider)
            
            # Count tokens for schema context
            schema_tokens = token_utils.count_tokens(schema_context_str, client.model_name, client.provider)
            
            # Count tokens for insights context
            insights_tokens = token_utils.count_tokens(insights_context_str, client.model_name, client.provider)
            
            # Count tokens for few-shot examples
            few_shot_tokens = token_utils.count_tokens(few_shot_examples_str, client.model_name, client.provider)
            
            # Track insights tokens separately
            client.token_tracker.add_insights_tokens(insights_tokens)
            
            # Calculate other prompt tokens (prompt tokens minus schema and insights tokens)
            other_prompt_tokens = prompt_tokens - schema_tokens - insights_tokens - few_shot_tokens
            if other_prompt_tokens < 0:
                other_prompt_tokens = 0
            
            # Track schema tokens and other prompt tokens separately
            client.token_tracker.add_tokens(0, 0, schema_tokens=schema_tokens, other_prompt_tokens=other_prompt_tokens)
            
            # Send the message to the LLM - this will also track tokens via client.token_tracker
            llm_response_obj = await client._send_message_to_llm(current_call_messages, natural_language_question, schema_tokens, source='sql_generation', command_type="generate_sql")
            # _send_message_to_llm adds the user prompt to client.conversation_history
            # _process_llm_response will add the assistant's response
            
            # _process_llm_response returns (text_content, tool_calls_made)
            # For this call, we expect tool_calls_made to be False.
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj, command_type="generate_sql")

            if tool_calls_made:
                last_error_feedback_to_llm = "Your response included an unexpected tool call. Please provide the JSON response directly."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM attempted tool call instead of providing JSON for SQL generation.")

            # print(f"LLM processed response for SQL gen (Attempt {attempt + 1}): {llm_response_text}") # Internal Detail

            json_from_response = _extract_json_from_response(llm_response_text)

            if not json_from_response:
                last_error_feedback_to_llm = "Your response did not contain a valid JSON object. Please provide the JSON output as instructed."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM failed to provide a JSON response for SQL generation.")

            parsed_response = SQLGenerationResponse.model_validate_json(json_from_response)
            
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
                        # --- Intelligent Retry: Fetch Correct Schema ---
                        detailed_schema_context = ""
                        try:
                            tables_in_query = _extract_tables_from_query(sql_to_execute)
                            if tables_in_query:
                                table_schemas = []
                                for table_name in tables_in_query:
                                    try:
                                        table_info_obj = await client.session.call_tool(
                                            "describe_table", {"table_name": table_name}
                                        )
                                        table_info = client._extract_mcp_tool_call_output(table_info_obj, "describe_table")
                                        table_schemas.append(json.dumps({table_name: table_info}, indent=2))
                                    except Exception as e_desc:
                                        handle_exception(e_desc, natural_language_question, {"context": f"Describing table {table_name} in retry loop"})
                                detailed_schema_context = "\n".join(table_schemas)
                        except Exception as e_extract:
                             handle_exception(e_extract, natural_language_question, {"context": "Extracting tables from query in retry loop"})
                        # --- End Intelligent Retry ---

                        fix_user_message_content = (
                            f"The previously generated SQL query resulted in an execution error.\n"
                            f"ORIGINAL NATURAL LANGUAGE QUESTION: \"{natural_language_question}\"\n\n"
                            f"SQL QUERY WITH ERROR:\n```sql\n{sql_to_execute}\n```\n\n"
                            f"EXECUTION ERROR MESSAGE:\n{execution_error}\n\n"
                            f"To help you fix this, here is the precise, correct schema for the table(s) involved in the query:\n"
                            f"CORRECT AND DETAILED SCHEMA:\n```json\n{detailed_schema_context}\n```\n\n"
                            f"Please provide a corrected SQL query using ONLY the columns listed in the schema above. For the explanation, describe how the *corrected* SQL query answers the original question. Do not mention the error or the process of fixing it.\n"
                            f"Respond ONLY with a single, valid JSON object matching this structure: "
                            f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}"
                        )
                        
                        fix_call_messages = client.conversation_history + [{"role": "user", "content": fix_user_message_content}]
                        
                        try:
                            # Count tokens for the complete prompt
                            fix_prompt_tokens = token_utils.count_tokens(fix_user_message_content, client.model_name, client.provider)
                            
                            # Count tokens for detailed schema context
                            schema_tokens = token_utils.count_tokens(detailed_schema_context, client.model_name, client.provider)
                            
                            # Calculate other prompt tokens
                            other_prompt_tokens = fix_prompt_tokens - schema_tokens
                            if other_prompt_tokens < 0:
                                other_prompt_tokens = 0
                                
                            # Track schema tokens and other prompt tokens separately
                            client.token_tracker.add_tokens(0, 0, schema_tokens=schema_tokens, other_prompt_tokens=other_prompt_tokens)
                            
                            fix_llm_response_obj = await client._send_message_to_llm(fix_call_messages, natural_language_question, schema_tokens, source='sql_generation', command_type="generate_sql")
                            fix_text, fix_tool_calls_made = await client._process_llm_response(fix_llm_response_obj, command_type="generate_sql")

                            if fix_tool_calls_made:
                                continue

                            json_from_fix = _extract_json_from_response(fix_text)
                            
                            if not json_from_fix:
                                continue

                            fixed_response = SQLGenerationResponse.model_validate_json(json_from_fix)
                            
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

async def regenerate_sql_query(
    client: 'LiteLLMMcpClient', 
    original_sql: str,
    user_feedback: str,
    schema_and_sample_data: Optional[Dict[str, Any]],
    insights_markdown_content: Optional[str],
    row_limit_for_preview: int = 1,
    command_type: str = "feedback" # Added for conversation history management
) -> Dict[str, Any]:
    """
    Regenerates an SQL query based on user feedback, using the HyDE approach.
    
    Args:
        client: The LiteLLMMcpClient instance.
        original_sql: The original SQL query that needs to be regenerated.
        user_feedback: The user's feedback on the original query.
        schema_and_sample_data: Dictionary containing schema and sample data for tables.
        insights_markdown_content: String content of the cumulative insights markdown file.
        row_limit_for_preview: Maximum number of rows to return in the preview.
        
    Returns:
        A dictionary containing the regenerated SQL query, explanation, execution results, and user message.
    """
    if not client.session or not client.current_db_name_identifier:
        return {
            "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
            "message_to_user": "Error: MCP session or DB identifier not available for SQL regeneration."
        }

    # --- HyDE: Retrieve Focused Schema Context ---
    hyde_context = ""
    table_names_from_hyde = []
    try:
        hyde_context, table_names_from_hyde = await hyde_regeneration_module.retrieve_hyde_regeneration_context(
            original_sql=original_sql,
            user_feedback=user_feedback,
            db_name_identifier=client.current_db_name_identifier,
            llm_client=client
        )
    except Exception as e_hyde:
        handle_exception(e_hyde, user_feedback, {"context": "HyDE Context Retrieval for Regeneration"})
        hyde_context = "Failed to retrieve focused schema context via HyDE for regeneration."
    # --- End HyDE ---

    # --- RAG: Retrieve Few-Shot Examples ---
    few_shot_examples_str = "No similar approved queries found to use as examples."
    try:
        current_rag_threshold = vector_store_module.LITELLM_RAG_THRESHOLD
        similar_pairs = vector_store_module.search_similar_nlqs(
            db_name_identifier=client.current_db_name_identifier,
            query_nlq=user_feedback,  # Use feedback as the query
            k=3,
            threshold=current_rag_threshold
        )
        if similar_pairs:
            examples_parts = ["Here are some examples of approved natural language questions and their corresponding SQL queries for this database:\n"]
            for i, pair in enumerate(similar_pairs):
                examples_parts.append(f"Example {i+1}:")
                examples_parts.append(f"  Natural Language Question: \"{pair['nlq']}\"")
                examples_parts.append(f"  SQL Query: ```sql\n{pair['sql']}\n```")
            few_shot_examples_str = "\n".join(examples_parts)
    except Exception as e_rag:
        handle_exception(e_rag, user_feedback, {"context": "RAG few-shot example retrieval for regeneration"})
    # --- End RAG ---

    # --- Load Schema Graph and Find Join Path ---
    schema_graph = memory_module.load_schema_graph(client.current_db_name_identifier)
    join_path_str = "No deterministic join path could be constructed for regeneration."
    if schema_graph and table_names_from_hyde:
        try:
            join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
            if join_clauses:
                join_path_str = "\n".join(join_clauses)
        except Exception as e_join:
            handle_exception(e_join, user_feedback, {"context": "Join Path Finder for Regeneration"})
            join_path_str = "Error constructing join path for regeneration."
    # --- End Join Path Finding ---

    # Prepare context strings for the prompt
    schema_context_str = "No schema or sample data provided."
    if schema_and_sample_data:
        try:
            # Full schema is kept as a fallback, but HyDE context is prioritized
            full_schema_str = json.dumps(schema_and_sample_data, indent=2)
            # If HyDE context is available and not an error message, use it. Otherwise, use full schema.
            if hyde_context and "Failed" not in hyde_context and "Error" not in hyde_context:
                schema_context_str = hyde_context
            else:
                schema_context_str = full_schema_str
        except TypeError as e:
            schema_context_str = f"Error serializing schema data: {e}. Data might be incomplete."
            handle_exception(e, user_feedback, {"context": "Serializing schema data for regeneration"})

    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content

    # Initial prompt for SQL regeneration
    current_prompt = (
        f"You are an expert PostgreSQL SQL writer. Based on the following database schema, sample data, "
        f"cumulative insights, the original SQL query, and the user's feedback, generate a new SQL query "
        f"(must start with SELECT) and a brief explanation.\n\n"
        f"ORIGINAL SQL QUERY:\n```sql\n{original_sql}\n```\n\n"
        f"USER FEEDBACK: \"{user_feedback}\"\n\n"
        f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the query):\n```\n{schema_context_str}\n```\n\n"
        f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{join_path_str}\n```\n\n"
        f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
        f"FEW-SHOT EXAMPLES (Use these to guide your SQL generation if relevant):\n{few_shot_examples_str}\n\n"
        f"IMPORTANT: Your response MUST be a single, valid JSON object. Do not include any other text, explanations, or markdown formatting outside of the JSON structure.\n"
        f"The JSON object must conform to this exact structure: "
        f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}"
    )

    llm_response_text = ""
    parsed_response: Optional[SQLGenerationResponse] = None
    last_error_feedback_to_llm = ""

    # Initial messages list for LiteLLM
    messages_for_llm = client.conversation_history[:]

    for attempt in range(MAX_SQL_GEN_RETRIES + 1):
        user_message_content = ""
        if attempt > 0 and last_error_feedback_to_llm:
            user_message_content = (
                f"{last_error_feedback_to_llm}\n\n"
                f"Based on the previous error, please try again.\n"
                f"ORIGINAL SQL QUERY:\n```sql\n{original_sql}\n```\n\n"
                f"USER FEEDBACK: \"{user_feedback}\"\n\n"
                f"DATABASE SCHEMA AND SAMPLE DATA:\n```json\n{schema_context_str}\n```\n\n"
                f"DETERMINISTIC JOIN PATH:\n```\n{join_path_str}\n```\n\n"
                f"CUMULATIVE INSIGHTS:\n```markdown\n{insights_context_str}\n```\n\n"
                f"FEW-SHOT EXAMPLES:\n{few_shot_examples_str}\n\n"
                f"Respond ONLY with a single JSON object matching the structure: "
                f"{{ \"sql_query\": \"<Your SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
                f"Ensure the SQL query strictly starts with 'SELECT'."
            )
        else:
            user_message_content = current_prompt

        current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]

        try:
            # Count tokens for the complete prompt
            prompt_text = user_message_content
            prompt_tokens = token_utils.count_tokens(current_prompt, client.model_name, client.provider)
            
            # Count tokens for schema context
            schema_tokens = token_utils.count_tokens(schema_context_str, client.model_name, client.provider)
            
            # Count tokens for insights context
            insights_tokens = token_utils.count_tokens(insights_context_str, client.model_name, client.provider)
            
            # Count tokens for few-shot examples
            few_shot_tokens = token_utils.count_tokens(few_shot_examples_str, client.model_name, client.provider)
            
            # Track insights tokens separately
            client.token_tracker.add_insights_tokens(insights_tokens)
            
            # Calculate other prompt tokens (prompt tokens minus schema and insights tokens)
            other_prompt_tokens = prompt_tokens - schema_tokens - insights_tokens - few_shot_tokens
            if other_prompt_tokens < 0:
                other_prompt_tokens = 0
            
            # Track schema tokens and other prompt tokens separately
            client.token_tracker.add_tokens(0, 0, schema_tokens=schema_tokens, other_prompt_tokens=other_prompt_tokens)
            
            # Send the message to the LLM - this will also track tokens via client.token_tracker
            llm_response_obj = await client._send_message_to_llm(current_call_messages, user_feedback, schema_tokens, source='sql_generation', command_type="feedback")
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj, command_type="feedback")

            if tool_calls_made:
                last_error_feedback_to_llm = "Your response included an unexpected tool call. Please provide the JSON response directly."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM attempted tool call instead of providing JSON for SQL regeneration.")

            json_from_response = _extract_json_from_response(llm_response_text)

            if not json_from_response:
                last_error_feedback_to_llm = "Your response did not contain a valid JSON object. Please provide the JSON output as instructed."
                if attempt < MAX_SQL_GEN_RETRIES: continue
                else: raise Exception("LLM failed to provide a JSON response for SQL regeneration.")

            parsed_response = SQLGenerationResponse.model_validate_json(json_from_response)
            
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
                    parsed_response = None
                    continue
                else: raise Exception("LLM failed to generate a SELECT query after retries.")
            
            break

        except Exception as e:
            context = {
                "context": "SQL regeneration loop",
                "attempt": attempt + 1,
                "llm_response_text": llm_response_text,
                "db_name_identifier": client.current_db_name_identifier
            }
            user_message = handle_exception(e, user_feedback, context)
            last_error_feedback_to_llm = f"A validation error occurred: {user_message}. Please try again."
            if attempt == MAX_SQL_GEN_RETRIES:
                return {
                    "sql_query": None, "explanation": None, "execution_result": None, "execution_error": user_message,
                    "message_to_user": "Failed to regenerate SQL query after all attempts."
                }

    if not parsed_response or not parsed_response.sql_query:
        return {
            "sql_query": None, "explanation": None, "execution_result": None, "execution_error": None,
            "message_to_user": "Failed to regenerate SQL query after all attempts."
        }

    sql_to_execute = parsed_response.sql_query
    explanation = parsed_response.explanation
    execution_result = None
    execution_error = None
    
    # Execute SQL with retry logic for execution errors
    original_regenerated_sql = sql_to_execute
    original_regenerated_explanation = explanation
    
    for exec_attempt in range(MAX_SQL_EXECUTION_RETRIES + 1):
        try:
            exec_result_obj = await client.session.call_tool(
                "execute_postgres_query",
                {"query": sql_to_execute, "row_limit": row_limit_for_preview}
            )
            raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj, "execute_postgres_query")

            try:
                if isinstance(raw_exec_output, str):
                    exec_data = json.loads(raw_exec_output)
                else:
                    exec_data = raw_exec_output

                if isinstance(exec_data, dict) and exec_data.get("status") == "error":
                    execution_error = exec_data.get("message", "Unknown execution error.")
                    
                    if exec_attempt < MAX_SQL_EXECUTION_RETRIES:
                        # --- HyDE for Error Correction ---
                        error_hyde_context = ""
                        error_table_names_from_hyde = []
                        try:
                            error_hyde_context, error_table_names_from_hyde = await hyde_regeneration_module.retrieve_hyde_regeneration_context(
                                original_sql=sql_to_execute,
                                user_feedback=f"Fix this error: {execution_error}",  # Use error as feedback
                                db_name_identifier=client.current_db_name_identifier,
                                llm_client=client
                            )
                        except Exception as e_hyde:
                            handle_exception(e_hyde, user_feedback, {"context": "HyDE Context Retrieval for Error Correction in Regeneration"})
                            error_hyde_context = "Failed to retrieve focused schema context via HyDE for error correction."
                        
                        # --- Load Schema Graph and Find Join Path for Error Correction ---
                        error_join_path_str = "No deterministic join path could be constructed for error correction."
                        if schema_graph and error_table_names_from_hyde:
                            try:
                                error_join_clauses = join_path_finder.find_join_path(error_table_names_from_hyde, schema_graph)
                                if error_join_clauses:
                                    error_join_path_str = "\n".join(error_join_clauses)
                            except Exception as e_join:
                                handle_exception(e_join, user_feedback, {"context": "Join Path Finder for Error Correction in Regeneration"})
                                error_join_path_str = "Error constructing join path for error correction."
                        
                        fix_prompt = (
                            f"The previously regenerated SQL query resulted in an execution error.\n"
                            f"ORIGINAL SQL QUERY:\n```sql\n{original_sql}\n```\n\n"
                            f"USER FEEDBACK: \"{user_feedback}\"\n\n"
                            f"REGENERATED SQL WITH ERROR:\n```sql\n{sql_to_execute}\n```\n\n"
                            f"EXECUTION ERROR MESSAGE:\n{execution_error}\n\n"
                            f"RELEVANT DATABASE SCHEMA INFORMATION:\n```\n{error_hyde_context}\n```\n\n"
                            f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{error_join_path_str}\n```\n\n"
                            f"Please provide a corrected SQL query that fixes the error while still addressing the user's feedback. "
                            f"For the explanation, describe how the *corrected* SQL query addresses the user's feedback. Do not mention the error or the fixing process.\n"
                            f"Respond ONLY with a single JSON object: {{ \"sql_query\": \"<corrected SELECT query>\", \"explanation\": \"<explanation>\" }}"
                        )
                        
                        fix_call_messages = client.conversation_history + [{"role": "user", "content": fix_prompt}]
                        
                        try:
                            # Count tokens for the complete prompt
                            fix_prompt_tokens = token_utils.count_tokens(fix_prompt, client.model_name, client.provider)
                            
                            # Count tokens for error schema context
                            schema_tokens = token_utils.count_tokens(error_hyde_context, client.model_name, client.provider)
                            
                            # Calculate other prompt tokens
                            other_prompt_tokens = fix_prompt_tokens - schema_tokens
                            if other_prompt_tokens < 0:
                                other_prompt_tokens = 0
                                
                            # Track schema tokens and other prompt tokens separately
                            client.token_tracker.add_tokens(0, 0, schema_tokens=schema_tokens, other_prompt_tokens=other_prompt_tokens)
                            
                            # Send the message to the LLM - this will also track tokens via client.token_tracker
                            fix_llm_response_obj = await client._send_message_to_llm(fix_call_messages, user_feedback, schema_tokens, source='sql_generation', command_type="feedback")
                            fix_text, _ = await client._process_llm_response(fix_llm_response_obj, command_type="feedback")

                            if fix_text.startswith("```json"): fix_text = fix_text[7:]
                            if fix_text.endswith("```"): fix_text = fix_text[:-3]
                            fix_text = fix_text.strip()

                            if fix_text:
                                fixed_response = SQLGenerationResponse.model_validate_json(fix_text)
                                if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                    sql_to_execute = fixed_response.sql_query
                                    explanation = fixed_response.explanation
                                    execution_error = None
                                    continue
                        except Exception as fix_e:
                            handle_exception(fix_e, user_query=user_feedback, context={"step": "fix_regenerated_sql_attempt", "attempt": exec_attempt + 1})
                else:
                    execution_result = exec_data
                    execution_error = None
                    break
            except json.JSONDecodeError:
                execution_result = raw_exec_output
                execution_error = None
                break
        except Exception as e:
            execution_error = handle_exception(e, user_query=user_feedback, context={"step": "execute_regenerated_sql_loop", "attempt": exec_attempt + 1})
            if exec_attempt >= MAX_SQL_EXECUTION_RETRIES:
                break
    
    if execution_error and sql_to_execute != original_regenerated_sql:
        sql_to_execute = original_regenerated_sql
        explanation = original_regenerated_explanation

    user_message = f"Regenerated SQL based on your feedback:\n```sql\n{sql_to_execute}\n```\n\nExplanation:\n{explanation}\n\n"
    if execution_error:
        user_message += f"Execution Error for regenerated SQL: {execution_error}\n"
    elif execution_result is not None:
        preview_str = ""
        if isinstance(execution_result, dict) and execution_result.get("status") == "success":
            data = execution_result.get("data")
            if data and isinstance(data, list) and len(data) > 0:
                preview_str = str(data[0])
            elif 'message' in execution_result:
                preview_str = execution_result['message']
            else:
                preview_str = "Query executed successfully, but no rows were returned."
        else:
            preview_str = str(execution_result)

        if len(preview_str) > 200:
            preview_str = preview_str[:197] + "..."
        user_message += f"Execution of regenerated SQL successful. Result preview (1 row): {preview_str}\n"
    
    user_message += "If this is correct, use '/approved'. If not, use '/feedback Your feedback text'."
    
    return {
        "sql_query": sql_to_execute,
        "explanation": explanation,
        "execution_result": execution_result,
        "execution_error": execution_error,
        "message_to_user": user_message
    }
