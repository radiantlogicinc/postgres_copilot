import json
import os
from typing import Tuple, Optional, Any, TYPE_CHECKING, Dict
import memory_module
from error_handler_module import handle_exception

if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient

def load_and_filter_schema(db_name_identifier: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Loads the master schema and filters it based on the active_tables.txt file.
    """
    # Read the master schema file
    schema_data_dict = memory_module.read_schema_data(db_name_identifier)
    if schema_data_dict is None:
        err_msg = "Schema file was not found or is empty. Cannot apply scope."
        return None, handle_exception(FileNotFoundError(err_msg), context={"step": "read_schema_file_for_filter"})

    # Create active_tables.txt if it doesn't exist
    active_tables_filepath = memory_module.get_active_tables_filepath(db_name_identifier)
    if not os.path.exists(active_tables_filepath):
        # Include both tables and views when creating the file for the first time
        all_object_names = [
            name for name, properties in schema_data_dict.items()
            if properties.get('type') in ('TABLE', 'VIEW')
        ]
        with open(active_tables_filepath, 'w') as f:
            for object_name in all_object_names:
                f.write(f"{object_name}\n")
        print(f"\n[INFO] A new file has been created to manage the database scope.")
        print(f"Please edit '{active_tables_filepath}' to add or remove tables and views for the Co-pilot to consider.")

    # Filter the schema based on active_tables.txt
    try:
        with open(active_tables_filepath, 'r') as f:
            active_tables = [line.strip() for line in f if line.strip()]
        
        # Filter the schema based on active_tables.txt for both tables and views
        filtered_schema = {}
        for item_name, properties in schema_data_dict.items():
            # Include both tables and views only if they are in the active_tables list
            if (properties.get('type') == 'TABLE' or properties.get('type') == 'VIEW') and item_name in active_tables:
                filtered_schema[item_name] = properties
        
        if not filtered_schema:
            err_msg = "No tables from 'active_tables.txt' were found in the database schema. Please check the file for valid table names."
            return None, handle_exception(ValueError(err_msg), context={"step": "filter_schema"})

        return filtered_schema, f"Scope loaded. The Co-pilot now has access to {len(filtered_schema)} tables."

    except FileNotFoundError:
        err_msg = "active_tables.txt not found. Please ensure it exists or re-initialize the connection."
        return None, handle_exception(FileNotFoundError(err_msg), context={"step": "filter_schema"})


async def perform_initialization(
    mcp_client_session_handler: 'LiteLLMMcpClient',
    connection_string: str,
    db_name_identifier: Optional[str] = "default_db"
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Connects to the DB, tells the server to fetch schema and save it to a file,
    then reads that file to load the schema.
    """
    if not hasattr(mcp_client_session_handler, 'session') or not mcp_client_session_handler.session:
        err_msg = "MCP session not available."
        return False, handle_exception(ValueError(err_msg), context={"step": "check_mcp_session"}), None
    
    session = mcp_client_session_handler.session

    # --- Step 1: Connect to PostgreSQL ---
    try:
        connect_result_obj = await session.call_tool("connect_to_postgres", {"connection_string": connection_string})
        connect_message = mcp_client_session_handler._extract_mcp_tool_call_output(connect_result_obj, "connect_to_postgres")
        if "Error:" in connect_message:
            err_msg = f"Failed to connect: {connect_message}"
            return False, handle_exception(ConnectionError(err_msg), user_query=connection_string, context={"step": "connect_to_postgres"}), None
    except Exception as e:
        return False, handle_exception(e, user_query=connection_string, context={"step": "connect_to_postgres"}), None

    # --- Step 2: Get Schema and Save to File via MCP tool ---
    schema_filepath = memory_module.get_schema_filepath(db_name_identifier)
    os.makedirs(os.path.dirname(schema_filepath), exist_ok=True)

    try:
        schema_result_obj = await session.call_tool(
            "get_schema_and_sample_data",
            {"output_file_path": schema_filepath}
        )
        response_str = mcp_client_session_handler._extract_mcp_tool_call_output(schema_result_obj, "get_schema_and_sample_data")
        
        response_data = json.loads(response_str)
        if response_data.get("status") == "error":
            error_msg = response_data.get("message", "Unknown error from server.")
            server_error = Exception(f"Server-side error during schema generation: {error_msg}")
            return False, handle_exception(server_error, user_query=connection_string, context={"step": "get_schema_and_sample_data"}), None
        
        # --- New: Scope Management ---
        filtered_schema, message = load_and_filter_schema(db_name_identifier)
        if filtered_schema is None:
            return False, message, None
        
        active_tables_filepath = memory_module.get_active_tables_filepath(db_name_identifier)
        
        success_message = (
            f"Successfully initialized '{db_name_identifier}'. {message}\n"
            f"To change this, please edit '{active_tables_filepath}' and run /reload_scope."
        )
        
        return True, success_message, filtered_schema

    except Exception as e:
        return False, handle_exception(e, user_query=connection_string, context={"step": "get_schema_and_sample_data"}), None
