import json
from typing import Optional, Any
from pydantic import ValidationError
from error_handler_module import handle_exception, display_message

# Assuming pydantic_models.py and memory_module.py are accessible
from pydantic_models import InsightsExtractionModel, FeedbackReportContentModel
import memory_module
import hyde_feedback_module
import join_path_finder
from token_utils import count_tokens

# This module will interact with the LLM (via postgres_copilot_chat.py's _send_message_to_llm method)
# So, the main function here will be called from postgres_copilot_chat.py,
# which has access to the LiteLLM client instance.

async def generate_and_update_insights(
    litellm_mcp_client_instance: Any, # Instance of LiteLLMMcpClient
    feedback_report_markdown_content: str,
    db_name_identifier: str,
    sql_query: str = None,  # Add SQL query parameter
    user_feedback: str = None  # Add user feedback parameter
) -> bool:
    """
    Processes a feedback report to extract insights and updates the
    insights file specific to the given db_name_identifier using LiteLLM.

    Args:
        litellm_mcp_client_instance: The instance of LiteLLMMcpClient.
        feedback_report_markdown_content: The string content of the new feedback report.
        db_name_identifier: The identifier for the current database.

    Returns:
        True if insights were successfully generated and saved, False otherwise.
    """
    # print(f"Starting insights generation and update process for database: {db_name_identifier}...") # User sees main wait message

    # 1. Read existing insights for the specific database
    existing_insights_md_content = memory_module.read_insights_file(db_name_identifier)
    if not existing_insights_md_content:
        existing_insights_md_content = ""

    # --- HyDE: Retrieve Focused Schema Context ---
    hyde_context = ""
    table_names_from_hyde = []
    if sql_query and user_feedback:
        try:
            hyde_context, table_names_from_hyde = await hyde_feedback_module.retrieve_hyde_feedback_context(
                sql_query=sql_query,
                user_feedback=user_feedback,
                db_name_identifier=db_name_identifier,
                llm_client=litellm_mcp_client_instance
            )
        except Exception as e_hyde:
            handle_exception(e_hyde, user_feedback, {"context": "HyDE Context Retrieval for Feedback"})
            hyde_context = "Failed to retrieve focused schema context via HyDE for feedback."
    # --- End HyDE ---

    # --- Load Schema Graph and Find Join Path ---
    schema_graph = memory_module.load_schema_graph(db_name_identifier)
    join_path_str = "No deterministic join path could be constructed for feedback."
    if schema_graph and table_names_from_hyde:
        try:
            join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
            if join_clauses:
                join_path_str = "\n".join(join_clauses)
        except Exception as e_join:
            handle_exception(e_join, user_feedback, {"context": "Join Path Finder for Feedback"})
            join_path_str = "Error constructing join path for feedback."
    # --- End Join Path Finding ---

    # 2. Prepare prompt for LLM with HyDE context
    insights_model_json_schema = InsightsExtractionModel.model_json_schema()

    prompt = f"""
You are an AI assistant tasked with analyzing a SQL query feedback report and updating an insights document for a specific database.

**Objective:**
Extract key learnings, schema details, query best practices, common errors, and database-specific insights from the provided "NEW FEEDBACK REPORT CONTENT" for the database '{db_name_identifier}'.
Then, intelligently merge these new insights into the "EXISTING INSIGHTS FOR '{db_name_identifier}'" structure.
The goal is to build a rich, evolving knowledge base for this specific database.

**RELEVANT DATABASE SCHEMA INFORMATION:**
```
{hyde_context}
```

**DETERMINISTIC JOIN PATH (If applicable):**
```
{join_path_str}
```

**Output Format:**
Your final output MUST be a single JSON object that strictly conforms to the following Pydantic model schema for `InsightsExtractionModel`:
```json
{json.dumps(insights_model_json_schema, indent=2)}
```

**Key Instructions for Merging and Extraction:**
- **Uniqueness:** When adding insights (which are typically strings in lists), ensure that identical insights are not duplicated within the same list.
- **Categorization:** Correctly place new insights into the appropriate categories defined by the schema (e.g., `schema_understanding`, `query_construction_best_practices`, `specific_database_insights_map`).
- **`specific_database_insights_map`:**
    - The current database being analyzed is "{db_name_identifier}".
    - New insights specific to this database should be added to the list associated with the key "{db_name_identifier}" within `specific_database_insights_map`.
    - If this key doesn't exist in the existing insights for this DB, create it.
    - This map within the model for *this specific database's insights file* should primarily, if not exclusively, contain insights for `{db_name_identifier}`. If merging from a very old global file, be mindful.
- **Completeness:** Your output JSON should represent the *complete, updated* insights structure for '{db_name_identifier}', incorporating both old (from this DB's file) and new information. Do not just output the new insights.
- **Title and Introduction:** The `title` and `introduction` in the output JSON should be appropriate for an insights document specifically for the database '{db_name_identifier}'. You can adapt the existing ones or generate new ones if the existing ones are too generic or from a previous global file. For example, title could be "Insights for {db_name_identifier}".

**EXISTING INSIGHTS FOR '{db_name_identifier}' (Markdown Format, may be empty if new DB):**
```markdown
{existing_insights_md_content}
```

**NEW FEEDBACK REPORT CONTENT (Markdown Format):**
```markdown
{feedback_report_markdown_content}
```

Now, based on both the existing insights and the new feedback report, generate the complete, updated JSON object conforming to the `InsightsExtractionModel` schema.
Your entire response should be ONLY this JSON object.
"""

    # 3. Call LLM (via the client instance)
    # This is a simplified loop; a more robust one would handle retries for validation.
    max_attempts = 2
    llm_response_json_str = None # To store the string from LLM
    last_error_for_retry_prompt = ""

    # Start with a copy of the client's conversation history
    messages_for_llm = litellm_mcp_client_instance.conversation_history[:]

    for attempt in range(max_attempts):
        # print(f"Attempting LLM call for insights extraction (Attempt {attempt + 1}/{max_attempts})...") # Internal Detail
        
        current_user_prompt_content = prompt
        if attempt > 0 and last_error_for_retry_prompt:
            current_user_prompt_content = (
                f"{last_error_for_retry_prompt}\n\nPlease try to generate the insights JSON again, adhering to the schema.\n"
                f"Original request details remain the same (existing insights, new feedback report, and db_name_identifier: {db_name_identifier}).\n"
                f"Respond ONLY with the complete, updated JSON object for `InsightsExtractionModel`."
            )
        
        # Use a temporary list for the call to avoid polluting main history with retries until success
        current_call_messages = messages_for_llm + [{"role": "user", "content": current_user_prompt_content}]

        try:
            # Count tokens for the prompt
            schema_tokens = 0  # No schema in this case
            
            # Call LLM using the client's method. We expect JSON, so no tools.
            # This will also track tokens via client.token_tracker
            llm_response_obj = await litellm_mcp_client_instance._send_message_to_llm(
                current_call_messages, 
                f"Insights generation for {db_name_identifier}",
                schema_tokens
            )
            # _send_message_to_llm adds the user prompt to client.conversation_history
            # _process_llm_response will add the assistant's response and return the text content
            
            llm_response_json_str, tool_calls_made = await litellm_mcp_client_instance._process_llm_response(llm_response_obj)

            if tool_calls_made:
                last_error_for_retry_prompt = "Your response included an unexpected tool call. Please provide the JSON response for insights directly."
                if attempt == max_attempts - 1:
                    return False
                continue
            
            cleaned_response_text = llm_response_json_str.strip()
            if cleaned_response_text.startswith("```json"):
                cleaned_response_text = cleaned_response_text[7:]
            if cleaned_response_text.endswith("```"):
                cleaned_response_text = cleaned_response_text[:-3]
            cleaned_response_text = cleaned_response_text.strip()

            if not cleaned_response_text:
                last_error_for_retry_prompt = "Your response was empty. Please provide the JSON output for insights."
                if attempt == max_attempts - 1:
                    return False
                continue
            
            llm_response_json_str = cleaned_response_text 
            
            updated_insights_model = InsightsExtractionModel.model_validate_json(llm_response_json_str)

            memory_module.save_or_update_insights(updated_insights_model, db_name_identifier)
            return True

        except Exception as e: 
            user_message = handle_exception(e, user_query=f"Insights generation for {db_name_identifier}", context={"attempt": attempt + 1, "llm_response_text": llm_response_json_str})
            last_error_for_retry_prompt = f"An error occurred: {user_message}"
            if attempt == max_attempts - 1:
                return False
            
    return False

if __name__ == '__main__':
    # This module is not meant to be run directly for full functionality
    # as it depends on an active LiteLLMMcpClient instance for LLM calls.
    display_message("insights_module.py - This module requires a LiteLLMMcpClient instance to be fully tested.", level="WARNING")
    
    # Example: Test reading insights (doesn't require LLM)
    # memory_module.ensure_memory_directories() # memory_module does this on import
    # existing_md = memory_module.read_insights_file()
    # if existing_md:
    #     print("\nSuccessfully read existing summarized_insights.md:")
    #     print(existing_md[:500] + "...")
    # else:
    #     print("\nsummarized_insights.md not found or is empty.")
    pass
