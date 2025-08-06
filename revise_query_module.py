import json
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from pydantic import BaseModel, ValidationError

from pydantic_models import (
    SQLRevisionResponse, 
    RevisionIteration, 
    NLQGenerationForRevisedSQLResponse,
    RevisionReportContentModel
)
from error_handler_module import handle_exception
import memory_module
import hyde_revision_module
import join_path_finder

if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient


MAX_REVISE_SQL_RETRIES = 5
MAX_NLQ_GEN_RETRIES = 2

async def handle_revise_query_iteration(
    client: 'LiteLLMMcpClient',
    user_revision_prompt: str,
    current_sql_to_revise: str,
    revision_history_for_context: List[Dict[str, str]], # List of {"role": "user/assistant", "content": ...}
    row_limit_for_preview: int = 1 # Added for controlling preview rows
) -> Dict[str, Any]:
    """
    Handles a single iteration of revising an SQL query based on user prompt.
    """
    if not client.session:
        return {
            "revised_sql_query": None, "revised_explanation": None,
            "message_to_user": "Error: MCP session not available for SQL revision."
        }

    # Context for the LLM
    history_str = "\n".join([f"{item['role']}: {item['content']}" for item in revision_history_for_context])

    # --- HyDE: Retrieve Focused Schema Context ---
    hyde_context = ""
    table_names_from_hyde = []
    try:
        hyde_context, table_names_from_hyde = await hyde_revision_module.retrieve_hyde_revision_context(
            current_sql=current_sql_to_revise,
            revision_prompt=user_revision_prompt,
            db_name_identifier=client.current_db_name_identifier,
            llm_client=client
        )
    except Exception as e_hyde:
        handle_exception(e_hyde, user_revision_prompt, {"context": "HyDE Context Retrieval for Revision"})
        hyde_context = "Failed to retrieve focused schema context via HyDE for revision."
    # --- End HyDE ---

    # --- Load Schema Graph and Find Join Path ---
    schema_graph = memory_module.load_schema_graph(client.current_db_name_identifier)
    join_path_str = "No deterministic join path could be constructed for revision."
    if schema_graph and table_names_from_hyde:
        try:
            join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
            if join_clauses:
                join_path_str = "\n".join(join_clauses)
        except Exception as e_join:
            handle_exception(e_join, user_revision_prompt, {"context": "Join Path Finder for Revision"})
            join_path_str = "Error constructing join path for revision."
    # --- End Join Path Finding ---

    prompt = (
        f"You are an expert PostgreSQL SQL reviser. The user wants to revise an existing SQL query.\n\n"
        f"CURRENT SQL QUERY TO REVISE:\n```sql\n{current_sql_to_revise}\n```\n\n"
        f"REVISION HISTORY (if any):\n{history_str}\n\n"
        f"USER'S LATEST REVISION REQUEST: \"{user_revision_prompt}\"\n\n"
        f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the revised query):\n```\n{hyde_context}\n```\n\n"
        f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{join_path_str}\n```\n\n"
        f"Based on the current SQL, the user's latest request, and the provided schema information, generate a revised SQL query (must start with SELECT) and a brief explanation of the changes or how the new query addresses the request.\n"
        f"Respond ONLY with a single JSON object matching this structure: "
        f"{{ \"sql_query\": \"<Your revised SELECT SQL query>\", \"explanation\": \"<Your explanation>\" }}\n"
        f"Ensure the SQL query strictly starts with 'SELECT'."
    )

    llm_response_text = ""
    parsed_response: Optional[SQLRevisionResponse] = None # To be defined in pydantic_models
    last_error_feedback_to_llm = ""
    
    messages_for_llm = client.conversation_history[:] # Start with base history

    for attempt in range(MAX_REVISE_SQL_RETRIES + 1):
        user_message_content = prompt
        if attempt > 0 and last_error_feedback_to_llm:
            user_message_content = (
                f"{last_error_feedback_to_llm}\n\nPlease try again.\n"
                f"CURRENT SQL TO REVISE:\n```sql\n{current_sql_to_revise}\n```\n"
                f"USER'S LATEST REVISION REQUEST: \"{user_revision_prompt}\"\n"
                f"Respond ONLY with the JSON: {{ \"sql_query\": \"...\", \"explanation\": \"...\" }}"
            )

        current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]

        try:
            llm_response_obj = await client._send_message_to_llm(current_call_messages, user_revision_prompt)
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_feedback_to_llm = "Error: Your response included an unexpected tool call."
                if attempt == MAX_REVISE_SQL_RETRIES: raise Exception("LLM tool call error in revision.")
                continue

            cleaned_llm_response_text = llm_response_text.strip()
            if cleaned_llm_response_text.startswith("```json"): cleaned_llm_response_text = cleaned_llm_response_text[7:]
            if cleaned_llm_response_text.endswith("```"): cleaned_llm_response_text = cleaned_llm_response_text[:-3]
            
            parsed_response = SQLRevisionResponse.model_validate_json(cleaned_llm_response_text.strip())
            # temp_parsed = json.loads(cleaned_llm_response_text.strip()) # No longer needed
            # parsed_response = SQLRevisionResponse(**temp_parsed) # No longer needed


            if not parsed_response.sql_query or not parsed_response.sql_query.strip().upper().startswith("SELECT"):
                last_error_feedback_to_llm = "Error: SQL query must start with SELECT and be provided."
                if attempt == MAX_REVISE_SQL_RETRIES: raise Exception("LLM failed to provide valid SELECT SQL for revision.")
                parsed_response = None
                continue
            break
        except Exception as e:
            user_message = handle_exception(e, user_query=user_revision_prompt, context={"step": "handle_revise_query_iteration", "attempt": attempt + 1, "llm_response_text": llm_response_text})
            last_error_feedback_to_llm = f"Error processing your response: {user_message}."
            if attempt == MAX_REVISE_SQL_RETRIES:
                return {
                    "revised_sql_query": None, "revised_explanation": None,
                    "message_to_user": f"Failed to revise SQL after multiple attempts. Last error: {user_message}"
                }
    
    if not parsed_response or not parsed_response.sql_query:
        return {
            "revised_sql_query": None, "revised_explanation": None,
            "message_to_user": "Failed to generate revised SQL query."
        }

    # Execute the revised SQL for verification
    sql_to_execute = parsed_response.sql_query
    explanation = parsed_response.explanation
    execution_result = None
    execution_error = None
    original_sql = sql_to_execute
    original_explanation = explanation

    for exec_attempt in range(MAX_REVISE_SQL_RETRIES + 1):
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
                    
                    if exec_attempt < MAX_REVISE_SQL_RETRIES:
                        # --- HyDE for Error Correction ---
                        error_hyde_context = ""
                        error_table_names_from_hyde = []
                        try:
                            error_hyde_context, error_table_names_from_hyde = await hyde_revision_module.retrieve_hyde_revision_context(
                                current_sql=sql_to_execute,
                                revision_prompt=f"Fix this error: {execution_error}",  # Use error as feedback
                                db_name_identifier=client.current_db_name_identifier,
                                llm_client=client
                            )
                        except Exception as e_hyde:
                            handle_exception(e_hyde, user_revision_prompt, {"context": "HyDE Context Retrieval for Error Correction"})
                            error_hyde_context = "Failed to retrieve focused schema context via HyDE for error correction."
                        
                        # --- Load Schema Graph and Find Join Path for Error Correction ---
                        error_join_path_str = "No deterministic join path could be constructed for error correction."
                        if schema_graph and error_table_names_from_hyde:
                            try:
                                error_join_clauses = join_path_finder.find_join_path(error_table_names_from_hyde, schema_graph)
                                if error_join_clauses:
                                    error_join_path_str = "\n".join(error_join_clauses)
                            except Exception as e_join:
                                handle_exception(e_join, user_revision_prompt, {"context": "Join Path Finder for Error Correction"})
                                error_join_path_str = "Error constructing join path for error correction."
                        
                        fix_prompt = (
                            f"The previously revised SQL query resulted in an execution error.\n"
                            f"CURRENT SQL WITH ERROR:\n```sql\n{sql_to_execute}\n```\n\n"
                            f"EXECUTION ERROR MESSAGE:\n{execution_error}\n\n"
                            f"USER'S REVISION REQUEST: \"{user_revision_prompt}\"\n\n"
                            f"RELEVANT DATABASE SCHEMA INFORMATION:\n```\n{error_hyde_context}\n```\n\n"
                            f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{error_join_path_str}\n```\n\n"
                            f"Please provide a corrected SQL query that fixes the error while still addressing the user's revision request. "
                            f"For the explanation, describe how the *corrected* SQL query addresses the user's request. Do not mention the error or the fixing process.\n"
                            f"Respond ONLY with a single JSON object: {{ \"sql_query\": \"<corrected SELECT query>\", \"explanation\": \"<explanation>\" }}"
                        )
                        
                        fix_call_messages = client.conversation_history + [{"role": "user", "content": fix_prompt}]
                        
                        try:
                            fix_llm_response_obj = await client._send_message_to_llm(fix_call_messages, user_revision_prompt)
                            fix_text, _ = await client._process_llm_response(fix_llm_response_obj)

                            if fix_text.startswith("```json"): fix_text = fix_text[7:]
                            if fix_text.endswith("```"): fix_text = fix_text[:-3]
                            fix_text = fix_text.strip()

                            if fix_text:
                                fixed_response = SQLRevisionResponse.model_validate_json(fix_text)
                                if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                    sql_to_execute = fixed_response.sql_query
                                    explanation = fixed_response.explanation
                                    execution_error = None
                                    continue
                        except Exception as fix_e:
                            handle_exception(fix_e, user_query=user_revision_prompt, context={"step": "fix_revised_sql_attempt", "attempt": exec_attempt + 1})
                else:
                    execution_result = exec_data
                    execution_error = None
                    break
            except json.JSONDecodeError:
                execution_result = raw_exec_output
                execution_error = None
                break
        except Exception as e:
            execution_error = handle_exception(e, user_query=user_revision_prompt, context={"step": "execute_revised_sql_loop", "attempt": exec_attempt + 1})
            if exec_attempt >= MAX_REVISE_SQL_RETRIES:
                break
    
    if execution_error and sql_to_execute != original_sql:
        sql_to_execute = original_sql
        explanation = original_explanation

    user_message = f"Revised SQL:\n```sql\n{sql_to_execute}\n```\n\nExplanation:\n{explanation}\n\n"
    if execution_error:
        user_message += f"Execution Error for revised SQL: {execution_error}\n"
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

        if len(preview_str) > 200: # Truncate if too long
            preview_str = preview_str[:197] + "..."
        user_message += f"Execution of revised SQL successful. Result preview (1 row): {preview_str}\n"
    
    user_message += "Use `/revise Your new prompt` to revise again, `/feedback Your feedback` if the query is incorrect, or `/approve_revision` to finalize and save."

    return {
        "revised_sql_query": sql_to_execute,
        "revised_explanation": explanation,
        "execution_result": execution_result,
        "execution_error": execution_error,
        "message_to_user": user_message
    }

async def generate_nlq_for_revised_sql(
    client: 'LiteLLMMcpClient',
    final_revised_sql: str,
    revision_report: 'RevisionReportContentModel' # To be defined in pydantic_models
) -> Dict[str, Any]:
    """
    Generates a natural language question for the final revised SQL,
    considering the entire revision history.
    """
    # Create a summary of the revision process for context
    revision_summary_parts = [f"The initial SQL for revision was:\n```sql\n{revision_report.initial_sql_for_revision}\n```"]
    for i, iteration in enumerate(revision_report.revision_iterations):
        revision_summary_parts.append(
            f"\nRevision Iteration {i+1}:\n"
            f"User Prompt: \"{iteration.user_revision_prompt}\"\n"
            f"Resulting SQL:\n```sql\n{iteration.revised_sql_attempt}\n```"
        )
    revision_summary_str = "\n".join(revision_summary_parts)

    prompt = (
        f"You are an AI assistant. Based on an initial SQL query and a series of user-driven revisions, a final SQL query has been produced.\n\n"
        f"REVISION PROCESS SUMMARY:\n{revision_summary_str}\n\n"
        f"FINAL REVISED SQL QUERY:\n```sql\n{final_revised_sql}\n```\n\n"
        f"Your task is to generate a single, concise natural language question that accurately represents what this FINAL REVISED SQL QUERY achieves, "
        f"taking into account the entire revision process that led to it. The question should be something a user might naturally ask to get this final SQL query as a result.\n"
        f"Respond ONLY with a single JSON object matching this structure: "
        f"{{ \"natural_language_question\": \"<Your generated NLQ>\", \"reasoning\": \"<Briefly, why you chose this NLQ based on the history>\" }}"
    )

    llm_response_text = ""
    parsed_response: Optional[NLQGenerationForRevisedSQLResponse] = None # To be defined
    last_error_feedback_to_llm = ""
    messages_for_llm = client.conversation_history[:]

    for attempt in range(MAX_NLQ_GEN_RETRIES + 1):
        user_message_content = prompt
        if attempt > 0 and last_error_feedback_to_llm:
            user_message_content = (
                f"{last_error_feedback_to_llm}\n\nPlease try again.\n"
                f"FINAL REVISED SQL QUERY:\n```sql\n{final_revised_sql}\n```\n"
                f"Respond ONLY with the JSON: {{ \"natural_language_question\": \"...\", \"reasoning\": \"...\" }}"
            )
        
        current_call_messages = messages_for_llm + [{"role": "user", "content": user_message_content}]
        try:
            llm_response_obj = await client._send_message_to_llm(current_call_messages, "generate_nlq_for_revised_sql")
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_feedback_to_llm = "Error: Your response included an unexpected tool call."
                if attempt == MAX_NLQ_GEN_RETRIES: raise Exception("LLM tool call error in NLQ generation.")
                continue
            
            cleaned_llm_response_text = llm_response_text.strip()
            if cleaned_llm_response_text.startswith("```json"): cleaned_llm_response_text = cleaned_llm_response_text[7:]
            if cleaned_llm_response_text.endswith("```"): cleaned_llm_response_text = cleaned_llm_response_text[:-3]

            parsed_response = NLQGenerationForRevisedSQLResponse.model_validate_json(cleaned_llm_response_text.strip())
            # temp_parsed = json.loads(cleaned_llm_response_text.strip()) # No longer needed
            # parsed_response = NLQGenerationForRevisedSQLResponse(**temp_parsed) # No longer needed

            if not parsed_response.natural_language_question:
                last_error_feedback_to_llm = "Error: 'natural_language_question' field was not provided."
                if attempt == MAX_NLQ_GEN_RETRIES: raise Exception("LLM failed to provide NLQ.")
                parsed_response = None
                continue
            break
        except Exception as e:
            user_message = handle_exception(e, user_query="generate_nlq_for_revised_sql", context={"attempt": attempt + 1, "llm_response_text": llm_response_text})
            last_error_feedback_to_llm = f"Error processing your response: {user_message}."
            if attempt == MAX_NLQ_GEN_RETRIES:
                return {
                    "generated_nlq": None, "reasoning": None,
                    "message_to_user": f"Failed to generate NLQ for revised SQL. Last error: {user_message}"
                }

    if not parsed_response or not parsed_response.natural_language_question:
        return {
            "generated_nlq": None, "reasoning": None,
            "message_to_user": "Failed to generate NLQ for revised SQL."
        }

    return {
        "generated_nlq": parsed_response.natural_language_question,
        "reasoning": parsed_response.reasoning,
        "message_to_user": f"Successfully generated NLQ for the final revised SQL."
    }
