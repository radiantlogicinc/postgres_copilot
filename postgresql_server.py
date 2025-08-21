import sys
import os
import logging
import psycopg2
import json
import traceback
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

# Import necessary components from the mcp SDK
from mcp.server.fastmcp import FastMCP

# Configure logging
log_dir = '/app/data/logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'postgres_server.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    filename=log_file_path,
    filemode='w'
)
logger = logging.getLogger(__name__)
mcp_logger = logging.getLogger('mcp')
mcp_logger.setLevel(logging.DEBUG)
if not mcp_logger.handlers:
    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    mcp_logger.addHandler(file_handler)

logger.info("PostgreSQL MCP Server script started.")

# Global variable to hold the database connection
db_connection = None

# Create a FastMCP server instance
logger.info("Creating FastMCP instance for PostgreSQL server...")
mcp = FastMCP("PostgreSQLServer")
logger.info("FastMCP instance created.")

@mcp.tool()
def connect_to_postgres(connection_string: str) -> str:
    """
    Connects to a PostgreSQL database using the provided connection string.
    Example connection string: "postgresql://user:password@host:port/dbname"
    """
    global db_connection
    logger.debug(f"Tool 'connect_to_postgres' called with connection_string: {connection_string}")
    try:
        if db_connection:
            db_connection.close()
            logger.info("Closed existing database connection.")
        db_connection = psycopg2.connect(connection_string)
        logger.info("Successfully connected to PostgreSQL database.")
        return json.dumps({"status": "success", "message": "Successfully connected to PostgreSQL database."})
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}", exc_info=True)
        db_connection = None
        return json.dumps({"status": "error", "message": f"Failed to connect to PostgreSQL. {e}"})

@mcp.tool()
def get_schema_and_sample_data(output_file_path: str) -> str:
    """
    Fetches schema and sample data, saves it to a file, and returns status.
    Prioritizes 'public' schema to handle large databases.
    """
    global db_connection
    logger.debug(f"Tool 'get_schema_and_sample_data' called. Output path: {output_file_path}")
    if not db_connection:
        logger.warning("No active database connection.")
        return json.dumps({"status": "error", "message": "Not connected to any database."})

    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            # Get all user tables and views, prioritizing 'public' schema
            cur.execute("""
                SELECT * FROM (
                    SELECT schemaname, tablename, 'TABLE' as type
                    FROM pg_catalog.pg_tables
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    UNION ALL
                    SELECT schemaname, viewname as tablename, 'VIEW' as type
                    FROM pg_catalog.pg_views
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ) as all_objects
                ORDER BY CASE WHEN schemaname = 'public' THEN 0 ELSE 1 END, tablename;
            """)
            all_objects = cur.fetchall()
            
            logger.debug(f"Found {len(all_objects)} tables and views. Processing all of them.")

            result = {}
            for row in all_objects:
                item_name = row['tablename']
                schema_name = row['schemaname']
                item_type = row['type']
                full_item_name = f"{schema_name}.{item_name}"
                
                item_info = {"type": item_type, "schema": "", "sample_data": []}

                if item_type == 'VIEW':
                    cur.execute("SELECT definition FROM pg_views WHERE schemaname = %s AND viewname = %s;", (schema_name, item_name))
                    view_definition = cur.fetchone()
                    if view_definition:
                        item_info["schema"] = f"CREATE VIEW {full_item_name} AS\n{view_definition['definition']}"
                else: # It's a table
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position;
                    """, (schema_name, item_name))
                    columns = cur.fetchall()

                    create_statement_parts = [f"CREATE TABLE {full_item_name} ("]
                    for col in columns:
                        part = f"  {col['column_name']} {col['data_type']}"
                        if col['is_nullable'] == 'NO': part += " NOT NULL"
                        if col['column_default'] is not None: part += f" DEFAULT {col['column_default']}"
                        create_statement_parts.append(part + ",")
                    if create_statement_parts[-1].endswith(","):
                        create_statement_parts[-1] = create_statement_parts[-1][:-1]
                    create_statement_parts.append(");")
                    item_info["schema"] = "\n".join(create_statement_parts)

                query = sql.SQL("SELECT * FROM {}.{} LIMIT 5;").format(
                    sql.Identifier(schema_name), sql.Identifier(item_name)
                )
                cur.execute(query)
                sample_data = [dict(row) for row in cur.fetchall()]
                item_info["sample_data"] = sample_data

                result[full_item_name] = item_info

            # Save the result directly to the specified file path
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, default=str) # Use default=str for data types like datetime

            logger.info(f"Successfully saved schema for {len(all_objects)} tables and views to {output_file_path}.")
            return json.dumps({
                "status": "success",
                "message": f"Schema for {len(all_objects)} tables and views saved successfully.",
                "objects_found": len(all_objects),
                "objects_processed": len(all_objects)
            })

    except Exception as e:
        logger.error(f"Error fetching schema and data: {e}", exc_info=True)
        # Format the full traceback to send back to the client
        tb_str = traceback.format_exc()
        return json.dumps({"status": "error", "message": f"Server-side exception: {str(e)}\nTraceback:\n{tb_str}"})

@mcp.tool()
def get_foreign_key_relationships() -> str:
    """
    Queries the information_schema to retrieve all foreign key relationships.
    Returns a JSON string with a list of relationship objects.
    """
    global db_connection
    logger.debug("Tool 'get_foreign_key_relationships' called.")
    if not db_connection:
        logger.warning("No active database connection for 'get_foreign_key_relationships'.")
        return json.dumps({"status": "error", "message": "Not connected to any database."})

    query = """
    SELECT
        tc.table_name AS fk_table,
        kcu.column_name AS fk_column,
        ccu.table_name AS pk_table,
        ccu.column_name AS pk_column
    FROM
        information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
          AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY';
    """
    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            relationships = cur.fetchall()
            logger.info(f"Successfully fetched {len(relationships)} foreign key relationships.")
            return json.dumps({"status": "success", "data": relationships}, default=str)
    except Exception as e:
        logger.error(f"Error fetching foreign key relationships: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Could not fetch foreign key relationships. {e}"})

@mcp.tool()
def get_stored_functions() -> str:
    """
    Fetches all stored functions from the database with their definitions and comments.
    Returns a JSON string with function names, arguments, return types, definitions, and comments.
    """
    global db_connection
    logger.debug("Tool 'get_stored_functions' called.")
    if not db_connection:
        logger.warning("No active database connection for 'get_stored_functions'.")
        return json.dumps({"status": "error", "message": "Not connected to any database."})

    query = """
    SELECT 
        n.nspname AS schema_name,
        p.proname AS function_name,
        pg_get_function_arguments(p.oid) AS function_arguments,
        t.typname AS return_type,
        pg_get_functiondef(p.oid) AS function_definition,
        d.description AS function_comment
    FROM 
        pg_proc p
        LEFT JOIN pg_namespace n ON p.pronamespace = n.oid
        LEFT JOIN pg_type t ON p.prorettype = t.oid
        LEFT JOIN pg_description d ON p.oid = d.objoid
    WHERE 
        n.nspname NOT IN ('pg_catalog', 'information_schema')
        AND p.prokind = 'f'  -- 'f' for regular functions
    ORDER BY 
        n.nspname, p.proname;
    """
    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            functions = cur.fetchall()
            logger.info(f"Successfully fetched {len(functions)} stored functions.")
            return json.dumps({"status": "success", "data": functions}, default=str)
    except Exception as e:
        logger.error(f"Error fetching stored functions: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Could not fetch stored functions. {e}"})

@mcp.tool()
def execute_postgres_query(query: str) -> str:
    """
    Executes a given SQL query on the currently connected PostgreSQL database.
    Requires a prior successful connection using 'connect_to_postgres'.
    For SELECT queries, it returns a JSON string of the rows.
    For other queries (INSERT, UPDATE, DELETE), it returns a JSON string with a success message and row count.
    """
    global db_connection
    logger.debug(f"Tool 'execute_postgres_query' called with query: {query[:100]}...") # Log truncated query
    if not db_connection:
        logger.warning("No active database connection for 'execute_postgres_query'.")
        return json.dumps({"status": "error", "message": "Not connected to any database. Please use 'connect_to_postgres' first."})

    try:
        with db_connection.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query)
            
            if cur.description:
                results = cur.fetchall()
                logger.info(f"Query executed successfully. Fetched {len(results)} rows.")
                db_connection.commit()
                return json.dumps({"status": "success", "data": results}, default=str)
            else:
                rowcount = cur.rowcount
                db_connection.commit()
                logger.info(f"Query executed successfully. {rowcount} rows affected.")
                return json.dumps({"status": "success", "message": f"Query executed successfully. {rowcount} rows affected."})
                
    except Exception as e:
        db_connection.rollback()
        logger.error(f"Error executing query: {e}", exc_info=True)
        return json.dumps({"status": "error", "message": f"Could not execute query. {e}"})

logger.info("PostgreSQL MCP tools defined.")

if __name__ == "__main__":
    logger.info("__main__ block started for PostgreSQL server.")
    if sys.platform == "win32":
        logger.info("Configuring stdout/stdin for win32.")
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
        logger.info("stdout/stdin configured for win32.")

    logger.info("Starting MCP PostgreSQL Server (stdio)...")
    print("Starting MCP PostgreSQL Server (stdio)...")

    try:
        mcp.run(transport='stdio')
    except Exception as e:
        logger.critical(f"Critical error running MCP PostgreSQL server: {e}", exc_info=True)
    finally:
        if db_connection:
            db_connection.close()
            logger.info("Closed database connection on server exit.")
        logger.info("MCP PostgreSQL server finished or exited.")
        logging.shutdown()
