import json
import os
from typing import Tuple, Optional, Any, TYPE_CHECKING, Dict, List
import memory_module
import schema_vectorization_module
import graph_generation_module
import vector_store_module
import config_manager
from error_handler_module import handle_exception

if TYPE_CHECKING:
    from postgres_copilot_chat import LiteLLMMcpClient

def check_schema_vectors_exist(db_name_identifier: str) -> bool:
    """
    Check if schema vectors already exist for the given database.
    """
    conn = vector_store_module._get_lancedb_connection()
    table_name = schema_vectorization_module.get_schema_table_name(db_name_identifier)
    
    if table_name in conn.table_names():
        table = conn.open_table(table_name)
        # Check if the table has data
        return len(table) > 0
    return False

def check_schema_graph_exists(db_name_identifier: str) -> bool:
    """
    Check if schema graph already exists for the given database.
    """
    schema_graph = memory_module.load_schema_graph(db_name_identifier)
    return schema_graph is not None and len(schema_graph.get("nodes", [])) > 0

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
        display_filepath = config_manager.translate_path_for_display(active_tables_filepath)
        print(f"\n[INFO] A new file has been created to manage the database scope.")
        print(f"Please edit '{display_filepath}' to add or remove tables and views for the Co-pilot to consider.")

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


async def process_stored_functions(
    mcp_client_session_handler: 'LiteLLMMcpClient',
    db_name_identifier: str
) -> Tuple[bool, str]:
    """
    Fetches stored functions from the database and adds them to the approved SQL JSON file and LanceDB database.
    
    Args:
        mcp_client_session_handler: The LiteLLM MCP client session handler
        db_name_identifier: Identifier for the database
        
    Returns:
        Tuple of (success, message)
    """
    if not hasattr(mcp_client_session_handler, 'session') or not mcp_client_session_handler.session:
        err_msg = "MCP session not available."
        return False, handle_exception(ValueError(err_msg), context={"step": "check_mcp_session"})
    
    session = mcp_client_session_handler.session
    
    try:
        # Fetch stored functions from the database
        functions_result_obj = await session.call_tool("get_stored_functions", {})
        response_str = mcp_client_session_handler._extract_mcp_tool_call_output(functions_result_obj, "get_stored_functions")
        
        response_data = json.loads(response_str)
        if response_data.get("status") == "error":
            error_msg = response_data.get("message", "Unknown error from server.")
            server_error = Exception(f"Server-side error during function retrieval: {error_msg}")
            return False, handle_exception(server_error, context={"step": "get_stored_functions"})
        
        functions = response_data.get("data", [])
        if not functions:
            return True, "No stored functions found in the database."
        
        functions_added = 0
        
        # Process each function and add it to the approved SQL JSON file and LanceDB database
        for function in functions:
            schema_name = function.get("schema_name", "public")
            function_name = function.get("function_name")
            function_arguments = function.get("function_arguments", "")
            function_definition = function.get("function_definition")
            function_comment = function.get("function_comment")
            
            if function_name and function_definition:
                # Generate a natural language description for the function
                # First try to use the function comment if available
                if function_comment:
                    nlq = function_comment
                else:
                    # Otherwise, generate a description from the function name
                    # Convert snake_case to natural language
                    words = function_name.split('_')
                    nlq = ' '.join(words).capitalize()
                    
                    # Add information about arguments if available
                    if function_arguments:
                        nlq += f" with parameters: {function_arguments}"
                
                # The SQL to execute the function
                # Format: SELECT * FROM schema_name.function_name(args)
                # For simplicity, we'll use a placeholder for arguments
                sql = f"SELECT * FROM {schema_name}.{function_name}(/* parameters here */);"
                
                # Add the NLQ-SQL pair to the approved SQL JSON file and LanceDB database
                try:
                    memory_module.save_nl2sql_pair(db_name_identifier, nlq, sql)
                    functions_added += 1
                except Exception as e:
                    handle_exception(e, context={"step": f"save_function_{function_name}"})
        
        return True, f"Successfully added {functions_added} stored functions to approved queries."
    
    except Exception as e:
        return False, handle_exception(e, context={"step": "process_stored_functions"})

async def perform_initialization(
    mcp_client_session_handler: 'LiteLLMMcpClient',
    connection_string: str,
    db_name_identifier: Optional[str] = "default_db",
    force_regenerate: bool = False
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Connects to the DB, tells the server to fetch schema and save it to a file,
    then reads that file to load the schema.
    
    Args:
        mcp_client_session_handler: The LiteLLM MCP client session handler
        connection_string: The PostgreSQL connection string
        db_name_identifier: Identifier for the database
        force_regenerate: If True, regenerate schema vectors and graph even if they exist
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

        # --- Check if schema vectors and graph already exist ---
        vectors_exist = check_schema_vectors_exist(db_name_identifier)
        graph_exists = check_schema_graph_exists(db_name_identifier)
        
        # --- New: Generate and Save Schema Graph (if needed) ---
        schema_graph = None
        foreign_key_relationships = None
        
        if not graph_exists or force_regenerate:
            try:
                print("[INFO] Generating schema relationship graph...")
                all_table_names = list(filtered_schema.keys())
                schema_graph = await graph_generation_module.generate_schema_graph(mcp_client_session_handler, all_table_names)
                if schema_graph:
                    memory_module.save_schema_graph(db_name_identifier, schema_graph)
                    print("[INFO] Schema relationship graph saved.")
                    
                    # Extract foreign key relationships from the schema graph
                    if 'edges' in schema_graph:
                        foreign_key_relationships = []
                        for edge in schema_graph['edges']:
                            source = edge.get('source')
                            target = edge.get('target')
                            label = edge.get('label')
                            
                            if source and target and label:
                                # Parse the label which is in format "fk_column → pk_column"
                                parts = label.split(' → ')
                                if len(parts) == 2:
                                    fk_column = parts[0]
                                    pk_column = parts[1]
                                    foreign_key_relationships.append({
                                        'fk_table': source,
                                        'fk_column': fk_column,
                                        'pk_table': target,
                                        'pk_column': pk_column
                                    })
            except Exception as e:
                handle_exception(e, context={"step": "generate_schema_graph"}, message="Could not generate schema graph, but continuing.")
        else:
            print("[INFO] Using existing schema graph.")
            # Load the existing schema graph to extract foreign key relationships
            schema_graph = memory_module.load_schema_graph(db_name_identifier)
            if schema_graph and 'edges' in schema_graph:
                foreign_key_relationships = []
                for edge in schema_graph['edges']:
                    source = edge.get('source')
                    target = edge.get('target')
                    label = edge.get('label')
                    
                    if source and target and label:
                        # Parse the label which is in format "fk_column → pk_column"
                        parts = label.split(' → ')
                        if len(parts) == 2:
                            fk_column = parts[0]
                            pk_column = parts[1]
                            foreign_key_relationships.append({
                                'fk_table': source,
                                'fk_column': fk_column,
                                'pk_table': target,
                                'pk_column': pk_column
                            })

        # --- New: Vectorize Schema for HyDE (if needed) ---
        if not vectors_exist or force_regenerate:
            try:
                print("[INFO] Vectorizing schema for enhanced search...")
                schema_vectorization_module.vectorize_and_store_schema(
                    db_name_identifier, 
                    filtered_schema, 
                    foreign_key_relationships,
                    force_regenerate=force_regenerate
                )
                print("[INFO] Schema vectorization complete.")
            except Exception as e:
                # Log the error but don't fail the entire initialization
                handle_exception(e, context={"step": "vectorize_schema"}, message="Could not vectorize schema, but continuing.")
        else:
            print("[INFO] Using existing schema vectors.")
        
        # --- Process stored functions (only when regenerating schema or changing database) ---
        if force_regenerate or not vectors_exist or not graph_exists:
            try:
                print("[INFO] Processing stored functions...")
                success, func_message = await process_stored_functions(mcp_client_session_handler, db_name_identifier)
                if success:
                    print(f"[INFO] {func_message}")
                else:
                    handle_exception(Exception(func_message), context={"step": "process_stored_functions"}, 
                                    message="Could not process stored functions, but continuing.")
            except Exception as e:
                handle_exception(e, context={"step": "process_stored_functions"}, 
                                message="Error processing stored functions, but continuing.")
        
        active_tables_filepath = memory_module.get_active_tables_filepath(db_name_identifier)
        display_filepath = config_manager.translate_path_for_display(active_tables_filepath)
        
        success_message = (
            f"Successfully initialized '{db_name_identifier}'. {message}\n"
            f"To change this, please edit '{display_filepath}' and run /reload_scope."
        )
        
        return True, success_message, filtered_schema

    except Exception as e:
        return False, handle_exception(e, user_query=connection_string, context={"step": "get_schema_and_sample_data"}), None
