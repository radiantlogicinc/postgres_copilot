import sys
import datetime
import json
from typing import TYPE_CHECKING, Optional, Dict, Any
import copy

from pydantic import ValidationError
import memory_module
from pydantic_models import InsightsExtractionModel, RevisionReportContentModel
from error_handler_module import handle_exception

if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient

# Define a constant for the maximum number of retries for LLM calls
MAX_LLM_RETRIES = 2

async def generate_insights_from_revision_history(
    client: 'LiteLLMMcpClient', 
    revision_report: RevisionReportContentModel, 
    db_name_identifier: str
) -> bool:
    """
    Generates insights by prompting an LLM to analyze a query revision history,
    and merges them with existing insights for the database.
    """
    # All top-level prints in this function are being removed as per user feedback.
    # Error prints to sys.stderr will be retained for actual issues.

    # 1. Prepare the revision history content for the LLM
    revision_summary_parts = [
        f"Initial SQL for Revision:\n```sql\n{revision_report.initial_sql_for_revision}\n```\n",
        "Revision Iterations:"
    ]
    if not revision_report.revision_iterations:
        revision_summary_parts.append("  (No explicit revision steps, initial SQL was directly approved or modified into final form without iterative prompts)")
    else:
        for i, iteration in enumerate(revision_report.revision_iterations):
            revision_summary_parts.append(f"  Iteration {i+1}:")
            revision_summary_parts.append(f"    User Prompt: \"{iteration.user_revision_prompt}\"")
            revision_summary_parts.append(f"    LLM's Revised SQL Attempt:\n```sql\n{iteration.revised_sql_attempt}\n```")
            revision_summary_parts.append(f"    LLM's Explanation: {iteration.revised_explanation or 'N/A'}")
    
    revision_summary_parts.append(f"\nFinal Approved SQL after revisions:\n```sql\n{revision_report.final_revised_sql_query}\n```")
    if revision_report.llm_generated_nlq_for_final_sql:
        revision_summary_parts.append(f"LLM-Generated Natural Language Question for Final SQL: \"{revision_report.llm_generated_nlq_for_final_sql}\"")
        if revision_report.llm_reasoning_for_nlq:
            revision_summary_parts.append(f"LLM's Reasoning for this NLQ: {revision_report.llm_reasoning_for_nlq}")

    revision_history_text = "\n".join(revision_summary_parts)

    # 2. Prompt LLM to extract new insights from this revision history
    
    # Constructing a more detailed guide for the LLM based on the nested structure
    # of InsightsExtractionModel and its sub-models.
    insights_guidance = (
        "Please populate the following fields within the main JSON object. "
        "If a specific list within a category has no new insights from this revision history, provide an empty list for it.\n\n"
        "1. `schema_understanding` (object):\n"
        "   - `table_structure_and_relationships` (list of strings): Observations about table structures and how they relate, learned or confirmed during the revisions.\n"
        "   - `field_knowledge` (list of strings): Specific knowledge about fields (columns), their meanings, data types, or typical values, learned or confirmed.\n\n"
        "2. `query_construction_best_practices` (object):\n"
        "   - `table_joins` (list of strings): Insights on effective or necessary table joins.\n"
        "   - `column_selection` (list of strings): Insights on selecting appropriate columns.\n"
        "   - `filtering_and_sorting` (list of strings): Insights on filtering (WHERE clauses) and sorting (ORDER BY clauses).\n"
        "   - `numeric_operations` (list of strings): Insights related to numeric operations or aggregations.\n\n"
        "3. `common_errors_and_corrections` (object):\n"
        "   - `schema_misunderstandings` (list of strings): Examples of schema misunderstandings that were corrected.\n"
        "   - `logical_errors` (list of strings): Examples of logical errors in query attempts that were corrected.\n"
        "   - `query_structure_issues` (list of strings): Issues related to query syntax or structure that were corrected.\n\n"
        "4. `data_validation_techniques` (object):\n"
        "   - `result_verification` (list of strings): Any implied methods or needs for verifying results that arose.\n"
        "   - `query_refinement` (list of strings): General observations about how queries were refined for correctness or clarity.\n\n"
        "5. `specific_database_insights_map` (object/dictionary where keys are database names and values are lists of strings):\n"
        "   - For this database, identified as '" + db_name_identifier + "', provide a list of specific insights under a key named exactly `\"" + db_name_identifier + "\"` if any unique conventions or tips were revealed. Example: `{\"" + db_name_identifier + "\": [\"Insight A for this DB\", \"Insight B for this DB\"]}`. If none, you can omit this key or provide an empty list for it.\n\n"
        "6. `general_sql_best_practices` (object):\n"
        "   - `practices` (list of strings): Broader SQL lessons or best practices applicable beyond this specific database, reinforced by the revision process.\n\n"
        "The top-level JSON should also include `title` and `introduction` with default or appropriate values."
    )

    insights_extraction_prompt = (
        f"You are an expert data analyst and PostgreSQL specialist. Analyze the following query revision history for the database '{db_name_identifier}'.\n"
        f"Your goal is to extract meaningful insights that can help improve future interactions with this database and structure them according to the `InsightsExtractionModel` Pydantic schema.\n\n"
        f"Revision History:\n{revision_history_text}\n\n"
        f"Based *only* on the information presented in this revision history, generate a single JSON object. "
        f"Follow this detailed guidance for populating the fields:\n{insights_guidance}\n\n"
        f"Full Pydantic Schema for `InsightsExtractionModel` for your reference (ensure your output strictly conforms to this overall structure, especially the nested objects for each category):\n"
        f"```json\n{json.dumps(InsightsExtractionModel.model_json_schema(), indent=2)}\n```\n"
        f"Respond ONLY with the single, complete, and valid JSON object. Ensure all specified list fields within the nested objects are present, even if empty."
    )
    
    extracted_insights_model: Optional[InsightsExtractionModel] = None
    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            messages_for_llm = [{"role": "user", "content": insights_extraction_prompt}]
            response_format = None
            if client.model_name.startswith("gpt-") or client.model_name.startswith("openai/"):
                response_format = {"type": "json_object"}

            # Count tokens for the prompt
            schema_tokens = 0  # No schema in this case
            
            # Send the message to the LLM - this will also track tokens via client.token_tracker
            llm_response_obj = await client._send_message_to_llm(
                messages=messages_for_llm, 
                user_query=f"Insights generation from revision for {db_name_identifier}",
                schema_tokens=schema_tokens,
                response_format=response_format
            )
            response_text, _ = await client._process_llm_response(llm_response_obj)

            if response_text.startswith("```json"): response_text = response_text[7:]
            if response_text.endswith("```"): response_text = response_text[:-3]
            
            # Validate against the full model, but we only prompted for a subset.
            # The model should handle default empty values for fields not prompted.
            extracted_insights_model = InsightsExtractionModel.model_validate_json(response_text.strip())
            break
        except Exception as e:
            user_message = handle_exception(e, user_query=f"Insights generation from revision for {db_name_identifier}", context={"attempt": attempt + 1})
            if attempt == MAX_LLM_RETRIES:
                return False
            insights_extraction_prompt += f"\nYour previous response was not valid. Error: {user_message}. Please try again."
            # No need to modify prompt for general errors, just retry.

    if not extracted_insights_model:
        return False

    # 3. Read existing insights
    existing_insights_md_content = memory_module.read_insights_file(db_name_identifier)
    if not existing_insights_md_content:
        existing_insights_md_content = ""
    
    # We'll create a new empty model for the LLM to populate
    # The LLM will use the markdown content directly in the prompt
    existing_insights_model = InsightsExtractionModel() # Create a new empty model

    # 4. Merge new insights with existing ones (LLM-assisted or rule-based)
    # For simplicity here, we'll do a basic merge. A more advanced LLM call could refine this.
    
    # Simple append/update merge strategy:
    final_insights_model = copy.deepcopy(existing_insights_model)

    # Helper function for merging lists within nested models
    def merge_list_field(final_category_obj, extracted_category_obj, field_name: str):
        extracted_list = getattr(extracted_category_obj, field_name, [])
        if extracted_list: # Only proceed if there's something to merge
            # Ensure the field exists in final_category_obj and is a list (should be by Pydantic default_factory)
            if not hasattr(final_category_obj, field_name):
                setattr(final_category_obj, field_name, []) # Should not be needed if Pydantic models are well-defined
            
            current_list = getattr(final_category_obj, field_name)
            if not isinstance(current_list, list): # Safeguard
                current_list = []
                setattr(final_category_obj, field_name, current_list)

            for item in extracted_list:
                if item not in current_list:
                    current_list.append(item)

    # Merge schema_understanding
    if hasattr(extracted_insights_model, 'schema_understanding'):
        extracted_su = extracted_insights_model.schema_understanding
        final_su = final_insights_model.schema_understanding # Relies on default_factory
        merge_list_field(final_su, extracted_su, 'table_structure_and_relationships')
        merge_list_field(final_su, extracted_su, 'field_knowledge')

    # Merge query_construction_best_practices
    if hasattr(extracted_insights_model, 'query_construction_best_practices'):
        extracted_qcbp = extracted_insights_model.query_construction_best_practices
        final_qcbp = final_insights_model.query_construction_best_practices
        merge_list_field(final_qcbp, extracted_qcbp, 'table_joins')
        merge_list_field(final_qcbp, extracted_qcbp, 'column_selection')
        merge_list_field(final_qcbp, extracted_qcbp, 'filtering_and_sorting')
        merge_list_field(final_qcbp, extracted_qcbp, 'numeric_operations')

    # Merge common_errors_and_corrections
    if hasattr(extracted_insights_model, 'common_errors_and_corrections'):
        extracted_cec = extracted_insights_model.common_errors_and_corrections
        final_cec = final_insights_model.common_errors_and_corrections
        merge_list_field(final_cec, extracted_cec, 'schema_misunderstandings')
        merge_list_field(final_cec, extracted_cec, 'logical_errors')
        merge_list_field(final_cec, extracted_cec, 'query_structure_issues')

    # Merge data_validation_techniques
    if hasattr(extracted_insights_model, 'data_validation_techniques'):
        extracted_dvt = extracted_insights_model.data_validation_techniques
        final_dvt = final_insights_model.data_validation_techniques
        merge_list_field(final_dvt, extracted_dvt, 'result_verification')
        merge_list_field(final_dvt, extracted_dvt, 'query_refinement')
        
    # Merge specific_database_insights_map
    if hasattr(extracted_insights_model, 'specific_database_insights_map'):
        for db_key, insights_list in extracted_insights_model.specific_database_insights_map.items():
            if insights_list: # Only if there are insights for this db_key
                if db_key not in final_insights_model.specific_database_insights_map:
                    final_insights_model.specific_database_insights_map[db_key] = []
                
                current_db_insights_list = final_insights_model.specific_database_insights_map[db_key]
                for item in insights_list:
                    if item not in current_db_insights_list:
                        current_db_insights_list.append(item)

    # Merge general_sql_best_practices
    if hasattr(extracted_insights_model, 'general_sql_best_practices'):
        extracted_gsbp = extracted_insights_model.general_sql_best_practices
        final_gsbp = final_insights_model.general_sql_best_practices
        merge_list_field(final_gsbp, extracted_gsbp, 'practices')
        
    # Update timestamp
    final_insights_model.last_updated = datetime.datetime.now().isoformat()


    # 5. Save the final merged insights
    try:
        memory_module.save_or_update_insights(final_insights_model, db_name_identifier)
        return True
    except Exception as e:
        handle_exception(e, user_query=f"Saving merged insights for {db_name_identifier}")
        return False
