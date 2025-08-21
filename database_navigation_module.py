import json
import re
import memory_module
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
import litellm
from error_handler_module import handle_exception, SchemaTooLargeError
from pydantic_models import SQLGenerationResponse
from pydantic import BaseModel, Field, ValidationError
import token_utils
import hyde_module
import sql_generation_module
if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient

# Pydantic model for LLM's decision on how to handle navigation
class NavigationAction(BaseModel):
    action_type: str = Field(description="Type of action to take: 'answer_directly'")
    explanation_for_user: Optional[str] = Field(None, description="Explanation or direct answer for the user.")

# Pydantic model for LLM's decision on how to handle database navigation with SQL
class NavigateDBAction(BaseModel):
    action_type: str = Field(description="Type of action to take: 'execute_sql', 'answer_directly', 'clarify_question'")
    sql_description: Optional[str] = Field(None, description="Description of what the SQL query should do if action_type is 'execute_sql'.")
    explanation_for_user: Optional[str] = Field(None, description="Explanation or direct answer for the user.")
    clarification_question_for_user: Optional[str] = Field(None, description="A question to ask the user for clarification.")

async def handle_navigation_query(
    client: 'LiteLLMMcpClient', 
    user_query: str,
    current_db_name: str, 
    insights_markdown_content: Optional[str]
) -> str:
    # Start token tracking for navigation query
    client.token_tracker.start_command("navigation")
    """
    Handles natural language queries for database schema exploration using LiteLLM.
    The LLM answers schema-related questions directly without executing SQL.
    """
    if not client.session: # For LiteLLM client, session is the key check for MCP
        return "Error: MCP session not available for database navigation."

    # Get schema backup content and table/view names
    database_context_str = f"The connected database is named '{current_db_name}'."
    try:
        # Try to read the schema backup file
        schema_backup_content = ""
        schema_backup_filepath = memory_module.get_schema_backup_filepath(client.current_db_name_identifier)
        
        if schema_backup_filepath.exists():
            with open(schema_backup_filepath, 'r', encoding='utf-8') as f:
                schema_backup_content = f.read().strip()
            # Use the schema backup content as the database context
            if schema_backup_content:
                database_context_str = schema_backup_content
        
        # Try to get table names and view names from the database
        schema_result_obj = await client.session.call_tool("get_schema_and_sample_data", {})
        tool_output = client._extract_mcp_tool_call_output(schema_result_obj, "get_schema_and_sample_data")
        
        if isinstance(tool_output, dict) and tool_output:
            table_names = [name for name, props in tool_output.items() if props.get('type') == 'TABLE']
            view_names = [name for name, props in tool_output.items() if props.get('type') == 'VIEW']
            
            context_parts = []
            if table_names:
                context_parts.append(f"Tables: {', '.join(table_names)}")
            if view_names:
                context_parts.append(f"Views: {', '.join(view_names)}")

            # Only use table/view names if we don't have schema backup content
            if not schema_backup_content and context_parts:
                database_context_str = (
                    f"The connected database ('{current_db_name}') has the following objects ({'; '.join(context_parts)}). "
                    f"No detailed schema information is available."
                )
    except Exception as e:
        handle_exception(e, user_query, {"detail": "Error fetching table names for navigation prompt context"})


    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content
    
    navigation_action_schema = NavigationAction.model_json_schema()

    prompt = (
        f"You are a database schema exploration assistant for a PostgreSQL database named '{current_db_name}'. "
        f"The user wants to understand the database schema with the following query: \"{user_query}\"\n\n"
        f"Available context:\n"
        f"1. Database Overview: {database_context_str}\n"
        f"2. Cumulative Insights (from previous interactions):\n```markdown\n{insights_context_str}\n```\n\n"
        f"Your task is to answer questions about the database schema ONLY. You should NOT generate or execute SQL queries. "
        f"You should only provide information about the database structure, tables, columns, relationships, and other schema-related information.\n\n"
        f"Always try to answer the user's question as best as you can based on the schema information available. "
        f"Even if the question is vague or ambiguous, provide the most helpful response possible with the information you have.\n\n"
        f"Respond ONLY with a single JSON object. "
        f"The JSON object MUST conform to the following structure (provide only the JSON, no other text before or after it, especially no ```json markdown backticks around the JSON output itself):\n"
        f"- `action_type` (string, required): Must be 'answer_directly'.\n"
        f"- `explanation_for_user` (string, required): Provide a detailed explanation about the database schema related to the user's query.\n\n"
        f"Example of a valid JSON response:\n"
        f"{{\n"
        f"  \"action_type\": \"answer_directly\",\n"
        f"  \"explanation_for_user\": \"The customers table has the following columns: id (integer, primary key), name (varchar), email (varchar), and created_at (timestamp).\"\n"
        f"}}\n"
        f"Guidelines for `NavigationAction` fields:\n"
        f"- `action_type`: Must be 'answer_directly'.\n"
        f"- `explanation_for_user`: Provide a detailed explanation about the database schema. If the user's query is vague, provide general information about the database structure that might be helpful.\n"
    )

    MAX_NAV_RETRIES = 1
    llm_response_text = ""
    last_error_for_retry_prompt = ""
    # Get navigation-specific conversation history
    messages_for_llm = client._get_context_for_command("navigation")

    for attempt in range(MAX_NAV_RETRIES + 1):
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to decide the navigation action again, adhering to the JSON schema.\n"
                f"Original user query: \"{user_query}\". Database: '{current_db_name}'.\n"
                f"Respond ONLY with the `NavigationAction` JSON object."
            )

        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]
        
        try:
            schema_tokens = token_utils.count_tokens(database_context_str, client.model_name, client.provider)
            llm_response_obj = await client._send_message_to_llm(current_call_messages, user_query, schema_tokens, command_type="navigation")
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj, command_type="navigation")

            if tool_calls_made:
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON decision directly."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM attempted tool call instead of providing JSON decision for navigation."
                continue

            if llm_response_text.startswith("```json"): llm_response_text = llm_response_text[7:]
            if llm_response_text.endswith("```"): llm_response_text = llm_response_text[:-3]
            llm_response_text = llm_response_text.strip()

            if not llm_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON decision."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM provided an empty response for navigation decision."
                continue
            
            action_decision = NavigationAction.model_validate_json(llm_response_text)

            # Get token usage for the navigation decision
            navigation_decision_token_usage = None
            
            # Note: 'execute_sql' action type has been removed from this function
            # Schema-related queries are now handled by 'answer_directly'

            if action_decision.action_type == 'answer_directly':
                # End token tracking
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigation query:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return (action_decision.explanation_for_user or "Based on the available schema information, I don't have enough details to answer your question specifically. Please try asking about specific tables or database structures.") + token_usage_message
            
            else:
                # End token tracking
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigation query:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return f"LLM chose an unknown or invalid action type: {action_decision.action_type}. Please try rephrasing.{token_usage_message}"

        except SchemaTooLargeError as e:
            handle_exception(e, user_query)
            # End token tracking on error
            token_usage = client.token_tracker.end_command()
            token_usage_message = ""
            if token_usage:
                token_usage_message = f"\n\nToken Usage for this navigation query:\n" \
                                     f"Input tokens: {token_usage['input_tokens']}\n" \
                                     f"Output tokens: {token_usage['output_tokens']}\n" \
                                     f"Total tokens: {token_usage['total_tokens']}\n" \
                                     f"Conversation total: {token_usage['conversation_total']} tokens"
            
            active_tables_filepath = memory_module.get_active_tables_filepath(client.current_db_name_identifier)
            return (
                f"The database schema is too large to fit into the model's context window. "
                f"Please reduce the number of active tables by editing the file at:\n"
                f"'{active_tables_filepath}'\n\n"
                f"Then, try your query again.{token_usage_message}"
            )

        except Exception as e:
            user_message = handle_exception(e, user_query, {"attempt": attempt + 1, "llm_response_text": llm_response_text})
            last_error_for_retry_prompt = f"An error occurred: {user_message}"
            if attempt == MAX_NAV_RETRIES:
                # End token tracking on final error
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigation query:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return user_message + token_usage_message
                
    # End token tracking if we somehow get here
    token_usage = client.token_tracker.end_command()
    token_usage_message = ""
    if token_usage:
        token_usage_message = f"\n\nToken Usage for this navigation query:\n" \
                             f"Input tokens: {token_usage['input_tokens']}\n" \
                             f"Output tokens: {token_usage['output_tokens']}\n" \
                             f"Total tokens: {token_usage['total_tokens']}\n" \
                             f"Conversation total: {token_usage['conversation_total']} tokens"
    
    return f"Failed to handle navigation query after all attempts.{token_usage_message}"

async def handle_navigate_db_command(
    client: 'LiteLLMMcpClient', 
    user_query: str,
    current_db_name: str, 
    insights_markdown_content: Optional[str]
) -> str:
    # Start token tracking for navigate_db command
    client.token_tracker.start_command("navigate_db")
    """
    Handles natural language queries for database navigation and exploration using LiteLLM.
    The LLM decides whether to execute a query, fetch schema, or answer directly.
    This function generates SQL to answer user questions and provides descriptive answers based on the data.
    """
    if not client.session: # For LiteLLM client, session is the key check for MCP
        return "Error: MCP session not available for database navigation."

    # Get schema backup content and table/view names
    database_context_str = f"The connected database is named '{current_db_name}'."
    try:
        # Try to read the schema backup file
        schema_backup_content = ""
        schema_backup_filepath = memory_module.get_schema_backup_filepath(client.current_db_name_identifier)
        
        if schema_backup_filepath.exists():
            with open(schema_backup_filepath, 'r', encoding='utf-8') as f:
                schema_backup_content = f.read().strip()
            # Use the schema backup content as the database context
            if schema_backup_content:
                database_context_str = schema_backup_content
        
        # Try to get table names and view names from the database
        schema_result_obj = await client.session.call_tool("get_schema_and_sample_data", {})
        tool_output = client._extract_mcp_tool_call_output(schema_result_obj, "get_schema_and_sample_data")
        
        if isinstance(tool_output, dict) and tool_output:
            table_names = [name for name, props in tool_output.items() if props.get('type') == 'TABLE']
            view_names = [name for name, props in tool_output.items() if props.get('type') == 'VIEW']
            
            context_parts = []
            if table_names:
                context_parts.append(f"Tables: {', '.join(table_names)}")
            if view_names:
                context_parts.append(f"Views: {', '.join(view_names)}")

            # Only use table/view names if we don't have schema backup content
            if not schema_backup_content and context_parts:
                database_context_str = (
                    f"The connected database ('{current_db_name}') has the following objects ({'; '.join(context_parts)}). "
                    f"No detailed schema information is available."
                )
    except Exception as e:
        handle_exception(e, user_query, {"detail": "Error fetching table names for navigation prompt context"})

    insights_context_str = "No cumulative insights provided."
    if insights_markdown_content and insights_markdown_content.strip():
        insights_context_str = insights_markdown_content
    
    navigate_db_action_schema = NavigateDBAction.model_json_schema()

    prompt = (
        f"You are a database exploration assistant for a PostgreSQL database named '{current_db_name}'. "
        f"The user wants to navigate or understand the database with the following query: \"{user_query}\"\n\n"
        f"Available context:\n"
        f"1. Database Overview: {database_context_str}\n"
        f"2. Cumulative Insights (from previous interactions):\n```markdown\n{insights_context_str}\n```\n\n"
        f"Your task is to decide the best course of action. You have access to the following tools:\n"
        f"- `execute_postgres_query(query: str)`: Executes a SQL query to fetch data from the database.\n\n"
        f"Based on the user's query, decide on an action. Respond ONLY with a single JSON object. "
        f"The JSON object MUST conform to the following structure (provide only the JSON, no other text before or after it, especially no ```json markdown backticks around the JSON output itself):\n"
        f"- `action_type` (string, required): Choose one of 'execute_sql', 'answer_directly', 'clarify_question'.\n"
        f"- `sql_description` (string, optional): If `action_type` is 'execute_sql', provide a description of what the SQL query should do. Do NOT write the actual SQL query.\n"
        f"- `explanation_for_user` (string, optional): If `action_type` is 'answer_directly', provide the answer. If 'execute_sql', briefly explain what you're about to do.\n"
        f"- `clarification_question_for_user` (string, optional): If `action_type` is 'clarify_question', provide the question to ask the user.\n\n"
        f"Example of a valid JSON response for 'execute_sql':\n"
        f"{{\n"
        f"  \"action_type\": \"execute_sql\",\n"
        f"  \"sql_description\": \"Find all records from the customers table where the ID equals 123\",\n"
        f"  \"explanation_for_user\": \"I will run a query to find the customer with ID 123.\"\n"
        f"}}\n"
        f"Example of a valid JSON response for 'clarify_question':\n"
        f"{{\n"
        f"  \"action_type\": \"clarify_question\",\n"
        f"  \"clarification_question_for_user\": \"Which specific columns are you interested in from the customer table?\"\n"
        f"}}\n"
        f"Guidelines for `NavigateDBAction` fields:\n"
        f"- `action_type`: Choose one of 'execute_sql', 'answer_directly', 'clarify_question'.\n"
        f"  - 'execute_sql': ONLY use this when the user is requesting very specific information that requires querying the database, such as details about a specific employee, product information, counts/statistics, or other precise data that cannot be answered from general knowledge or the provided context. Examples: 'Show me employee John Smith's details', 'How many products are in the inventory?', 'What's the total sales for last month?'\n"
        f"  - 'answer_directly': If the query is a general question that can be answered from the provided insights, database overview or general knowledge without hitting the database.\n"
        f"  - 'clarify_question': If the user's query is too vague or ambiguous to proceed.\n"
        f"- `sql_description`: If `action_type` is 'execute_sql', provide a description of what the SQL query should do. Do NOT write the actual SQL query.\n"
        f"- `explanation_for_user`: If `action_type` is 'answer_directly', provide the answer. If 'execute_sql', briefly explain what you're about to do.\n"
        f"- `clarification_question_for_user`: If `action_type` is 'clarify_question', provide the question to ask the user.\n"
    )

    MAX_NAV_RETRIES = 1
    llm_response_text = ""
    last_error_for_retry_prompt = ""
    # Get navigation-specific conversation history
    messages_for_llm = client._get_context_for_command("navigation")

    for attempt in range(MAX_NAV_RETRIES + 1):
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to decide the navigation action again, adhering to the JSON schema.\n"
                f"Original user query: \"{user_query}\". Database: '{current_db_name}'.\n"
                f"Respond ONLY with the `NavigateDBAction` JSON object."
            )

        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]
        
        try:
            schema_tokens = token_utils.count_tokens(database_context_str, client.model_name, client.provider)
            llm_response_obj = await client._send_message_to_llm(current_call_messages, user_query, schema_tokens, command_type="navigation")
            llm_response_text, tool_calls_made = await client._process_llm_response(llm_response_obj, command_type="navigation")

            if tool_calls_made:
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON decision directly."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM attempted tool call instead of providing JSON decision for navigation."
                continue

            if llm_response_text.startswith("```json"): llm_response_text = llm_response_text[7:]
            if llm_response_text.endswith("```"): llm_response_text = llm_response_text[:-3]
            llm_response_text = llm_response_text.strip()

            if not llm_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON decision."
                if attempt == MAX_NAV_RETRIES:
                    return "Error: LLM provided an empty response for navigation decision."
                continue
            
            action_decision = NavigateDBAction.model_validate_json(llm_response_text)

            # Get token usage for the navigation decision
            navigation_decision_token_usage = None
            
            if action_decision.action_type == 'execute_sql':
                if not action_decision.sql_description:
                    return "LLM decided to execute SQL but did not provide a description of what the SQL should do. Please try rephrasing."
                
                user_feedback_header = f"{action_decision.explanation_for_user or 'Generating SQL based on your request...'}"
                
                # Use hyde_module to get relevant schema context
                try:
                    
                    # Generate SQL using sql_generation_module
                    sql_gen_result = await sql_generation_module.generate_sql_query(
                        client, 
                        user_query, 
                        None,  # No need to pass schema_and_sample_data
                        client.cumulative_insights_content,
                        row_limit_for_preview=5,  # Limit to 5 rows as per requirements
                        command_type="navigation"  # Use navigation command type to maintain proper conversation history
                    )
                    
                    if not sql_gen_result.get("sql_query"):
                        return f"{user_feedback_header}\nFailed to generate SQL query: {sql_gen_result.get('message_to_user', 'Unknown error')}"
                    
                    generated_sql = sql_gen_result["sql_query"]
                    
                    user_feedback_header = (
                        f"{user_feedback_header}\n"
                        f"Executing SQL:\n```sql\n{generated_sql}\n```\n"
                    )
                    
                    # Execute the generated SQL
                    exec_result_obj = await client.session.call_tool("execute_postgres_query", {"query": generated_sql, "row_limit": 5})  # Set row limit to 5
                    raw_exec_output = client._extract_mcp_tool_call_output(exec_result_obj, "execute_postgres_query")
                except Exception as e_tool_call:
                    return f"{user_feedback_header}\n{handle_exception(e_tool_call, user_query, {'tool': 'execute_postgres_query'})}"

                # Handle errors first
                if isinstance(raw_exec_output, str) and "Error:" in raw_exec_output:
                    return f"{user_feedback_header}\nExecution Error: {raw_exec_output}"

                # Handle no results
                is_empty_result = False
                if isinstance(raw_exec_output, str) and "No results found" in raw_exec_output:
                    is_empty_result = True
                elif isinstance(raw_exec_output, list) and not raw_exec_output:
                    is_empty_result = True
                elif isinstance(raw_exec_output, dict) and not raw_exec_output.get("data"):
                    is_empty_result = True

                if is_empty_result:
                    return f"{user_feedback_header}\nExecution Result: Query executed successfully. No results found."

                # If we have results, summarize them
                try:
                    result_json_str = json.dumps(raw_exec_output, indent=2, default=str)
                    
                    summarization_prompt = (
                        f"You are a helpful database assistant. The user asked: \"{user_query}\".\n"
                        f"To answer this, the following SQL query was executed:\n```sql\n{generated_sql}\n```\n"
                        f"The query returned this JSON data, which is a list of items:\n```json\n{result_json_str}\n```\n\n"
                        f"Your task is to summarize **all** of the items in this list in a friendly, natural language paragraph. "
                        f"Do not just summarize the first item. Mention several of the items to give a comprehensive overview. "
                        f"For example, instead of just describing the first product, you could say 'I found several products, including the Corner Desk, a Bolt, and a Cabinet with Doors.'\n"
                        f"Explain what the data represents in the context of the user's question. Be concise and helpful.\n"
                        f"Respond ONLY with the natural language summary. Do not include any other text or markdown."
                    )
                    
                    summary_messages = [{"role": "user", "content": summarization_prompt}]
                    summary_response_obj = await client._send_message_to_llm(summary_messages, user_query)
                    summary_text, _ = await client._process_llm_response(summary_response_obj)
                    
                    # End token tracking and get usage
                    token_usage = client.token_tracker.end_command()
                    token_usage_message = ""
                    if token_usage:
                        token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                             f"Input tokens: {token_usage['input_tokens']}\n" \
                                             f"Output tokens: {token_usage['output_tokens']}\n" \
                                             f"Total tokens: {token_usage['total_tokens']}\n" \
                                             f"Conversation total: {token_usage['conversation_total']} tokens"
                    
                    return f"{user_feedback_header}\n{summary_text}{token_usage_message}"

                except Exception as e_summarize:
                    # End token tracking even on error
                    token_usage = client.token_tracker.end_command()
                    token_usage_message = ""
                    if token_usage:
                        token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                             f"Input tokens: {token_usage['input_tokens']}\n" \
                                             f"Output tokens: {token_usage['output_tokens']}\n" \
                                             f"Total tokens: {token_usage['total_tokens']}\n" \
                                             f"Conversation total: {token_usage['conversation_total']} tokens"
                    
                    return f"{user_feedback_header}\n{handle_exception(e_summarize, user_query, {'detail': 'Failed to generate natural language summary.'})}\nRaw Result:\n{json.dumps(raw_exec_output, indent=2, default=str)}{token_usage_message}"

            elif action_decision.action_type == 'answer_directly':
                # End token tracking
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return (action_decision.explanation_for_user or "I'm not sure how to answer that directly without more specific database interaction.") + token_usage_message

            elif action_decision.action_type == 'clarify_question':
                # End token tracking
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return (action_decision.clarification_question_for_user or "Could you please clarify your request?") + token_usage_message
            
            else:
                # End token tracking
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return f"LLM chose an unknown or invalid action type: {action_decision.action_type}. Please try rephrasing.{token_usage_message}"

        except SchemaTooLargeError as e:
            handle_exception(e, user_query)
            # End token tracking on error
            token_usage = client.token_tracker.end_command()
            token_usage_message = ""
            if token_usage:
                token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                     f"Input tokens: {token_usage['input_tokens']}\n" \
                                     f"Output tokens: {token_usage['output_tokens']}\n" \
                                     f"Total tokens: {token_usage['total_tokens']}\n" \
                                     f"Conversation total: {token_usage['conversation_total']} tokens"
            
            active_tables_filepath = memory_module.get_active_tables_filepath(client.current_db_name_identifier)
            return (
                f"The database schema is too large to fit into the model's context window. "
                f"Please reduce the number of active tables by editing the file at:\n"
                f"'{active_tables_filepath}'\n\n"
                f"Then, try your query again.{token_usage_message}"
            )

        except Exception as e:
            user_message = handle_exception(e, user_query, {"attempt": attempt + 1, "llm_response_text": llm_response_text})
            last_error_for_retry_prompt = f"An error occurred: {user_message}"
            if attempt == MAX_NAV_RETRIES:
                # End token tracking on final error
                token_usage = client.token_tracker.end_command()
                token_usage_message = ""
                if token_usage:
                    token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                                         f"Input tokens: {token_usage['input_tokens']}\n" \
                                         f"Output tokens: {token_usage['output_tokens']}\n" \
                                         f"Total tokens: {token_usage['total_tokens']}\n" \
                                         f"Conversation total: {token_usage['conversation_total']} tokens"
                
                return user_message + token_usage_message
                
    # End token tracking if we somehow get here
    token_usage = client.token_tracker.end_command()
    token_usage_message = ""
    if token_usage:
        token_usage_message = f"\n\nToken Usage for this navigate_db command:\n" \
                             f"Input tokens: {token_usage['input_tokens']}\n" \
                             f"Output tokens: {token_usage['output_tokens']}\n" \
                             f"Total tokens: {token_usage['total_tokens']}\n" \
                             f"Conversation total: {token_usage['conversation_total']} tokens"
    
    return f"Failed to handle navigate_db command after all attempts.{token_usage_message}"
