import json
import os
from pathlib import Path
import appdirs
import getpass # For securely getting API keys if possible, or just input()
from error_handler_module import handle_exception

APP_NAME = "PostgresCopilot"
APP_AUTHOR = "PostgresCopilotTeam" # Or your actual author/org name

CONFIG_FILE_NAME = "config.json"

# --- Path Management ---

def get_config_dir() -> Path:
    """
    Gets the configuration directory.
    - If POSTGRES_COPILOT_DATA_DIR is set (in Docker), it uses {DATA_DIR}/config.
    - Otherwise, it falls back to the user's standard config directory.
    """
    docker_data_dir_str = os.environ.get('POSTGRES_COPILOT_DATA_DIR')
    if docker_data_dir_str:
        return Path(docker_data_dir_str) / "config"
    else:
        # Fallback for non-Docker execution
        return Path(appdirs.user_config_dir(APP_NAME, APP_AUTHOR))

def get_config_file_path() -> Path:
    """Gets the full path to the configuration file."""
    return get_config_dir() / CONFIG_FILE_NAME

def get_default_data_dir() -> Path:
    """
    Gets the default data directory.
    - If POSTGRES_COPILOT_DATA_DIR is set (in Docker), it uses that path.
    - Otherwise, it falls back to the user's standard data directory.
    """
    docker_data_dir_str = os.environ.get('POSTGRES_COPILOT_DATA_DIR')
    if docker_data_dir_str:
        return Path(docker_data_dir_str)
    else:
        # Fallback for non-Docker execution
        return Path(appdirs.user_data_dir(APP_NAME, APP_AUTHOR))

def get_sentence_transformer_cache_dir() -> Path:
    """
    Gets the cache directory for sentence transformer models.
    It is a subdirectory within the main data directory.
    """
    cache_dir = get_default_data_dir() / "sentence_transformer_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# --- Configuration Loading and Saving ---

def load_config() -> dict:
    """Loads the configuration from the JSON file."""
    config_path = get_config_file_path()
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            handle_exception(e, user_query="load_config")
            print(f"Error: Configuration file at {config_path} is corrupted or unreadable. Please delete it and re-run.")
            return {} # Return empty or raise an error
    return {}

def save_config(config_data: dict) -> None:
    """Saves the configuration to the JSON file."""
    try:
        config_dir = get_config_dir()
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = get_config_file_path()
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"Configuration saved to current_working_directory/data/config/config.json")
    except IOError as e:
        handle_exception(e, user_query="save_config")
        print(f"Error: Could not save configuration file to {get_config_file_path()}. Please check permissions.")

# --- Initial Setup ---

def initial_setup() -> dict:
    """Creates a default config.json and guides the user to edit it."""
    config_path = get_config_file_path()
    print("Welcome to PostgreSQL Co-Pilot Setup!")
    print("------------------------------------")
    print(f"A new configuration file has been created at: current_working_directory/data/config/config.json")
    print("Please open this file, fill in your details, and then type 'done' here to continue.")

    default_memory_path = get_default_data_dir() / "memory"
    default_approved_queries_path = get_default_data_dir() / "Approved_NL2SQL_Pairs"

    demo_config = {
        "llm_profiles": {
            "my_gemini_profile": {
                "provider": "gemini",
                "model_id": "gemini-2.5-pro",
                "credentials": {
                    "api_key": "YOUR_GEMINI_API_KEY"
                }
            }
        },
        "active_llm_profile_alias": "my_gemini_profile",
        "database_connections": {
            "my_local_db": "postgresql://user:password@host.docker.internal:5432/database_name"
        },
        "active_database_alias": "my_local_db",
        "memory_base_dir": str(default_memory_path),
        "approved_queries_dir": str(default_approved_queries_path),
        "nl2sql_vector_store_base_dir": str(default_memory_path / "lancedb_stores")
    }
    
    save_config(demo_config)
    
    # This function will now just create the file and return the demo data.
    # The actual waiting for "done" will be handled in the main chat loop.
    return demo_config

def _set_env_vars_from_profile(profile_data: dict):
    """Sets environment variables for LiteLLM from a given profile dictionary."""
    provider = profile_data.get("provider")
    credentials = profile_data.get("credentials", {})
    
    if provider == "openai":
        os.environ["OPENAI_API_KEY"] = credentials.get("api_key", "")
    elif provider == "gemini":
        os.environ["GEMINI_API_KEY"] = credentials.get("api_key", "")
    elif provider == "anthropic":
        os.environ["ANTHROPIC_API_KEY"] = credentials.get("api_key", "")
    elif provider == "bedrock":
        if credentials.get("aws_access_key_id"):
            os.environ["AWS_ACCESS_KEY_ID"] = credentials["aws_access_key_id"]
        if credentials.get("aws_secret_access_key"):
            os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["aws_secret_access_key"]
        if credentials.get("aws_region_name"):
            os.environ["AWS_REGION_NAME"] = credentials["aws_region_name"]
    elif provider == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = credentials.get("api_key", "")
    elif provider == "openrouter":
        os.environ["OPENROUTER_API_KEY"] = credentials.get("api_key", "")
        if credentials.get("openrouter_api_base_url"):
            os.environ["OPENROUTER_API_BASE"] = credentials["openrouter_api_base_url"]

# --- Main Configuration Accessor ---

def get_app_config() -> dict:
    """
    Gets the application configuration, handling the multi-profile structure.
    If config file doesn't exist, runs initial setup.
    If multiple profiles exist, it prompts the user to select one.
    Returns a flattened dictionary with the active profile's settings at the top level.
    """
    config = load_config()
    if not config:
        print("No configuration found. Starting initial setup...")
        config = initial_setup()

    # --- Profile Selection ---
    profiles = config.get("llm_profiles", {})
    active_alias = config.get("active_llm_profile_alias")

    # Prompt for profile selection only if there are multiple profiles AND
    # no valid active profile is already set.
    if len(profiles) > 1 and (not active_alias or active_alias not in profiles):
        print("Multiple LLM profiles found. Please choose one to use for this session:")
        profile_aliases = list(profiles.keys())
        for i, alias in enumerate(profile_aliases):
            print(f"{i+1}. {alias}")
        
        choice = -1
        while choice < 1 or choice > len(profile_aliases):
            try:
                raw_choice = input(f"Enter your choice (1-{len(profile_aliases)}): ")
                choice = int(raw_choice)
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        active_alias = profile_aliases[choice - 1]
        config["active_llm_profile_alias"] = active_alias
        save_config(config) # Save the choice for next time
        print(f"Active profile set to '{active_alias}'.")

    active_alias = config.get("active_llm_profile_alias")
    active_profile = config.get("llm_profiles", {}).get(active_alias)

    if not active_profile:
        print(f"Error: Active LLM profile '{active_alias}' not found in config. Please check config.json.")
        # Fallback to initial setup or exit
        print("Restarting setup...")
        config = initial_setup()
        active_alias = config.get("active_llm_profile_alias")
        active_profile = config.get("llm_profiles", {}).get(active_alias)
        if not active_profile:
            print("Setup failed. Exiting.")
            exit(1) # Or handle more gracefully

    # Create a flattened config for the application to use
    app_config = {
        "memory_base_dir": config.get("memory_base_dir"),
        "approved_queries_dir": config.get("approved_queries_dir"),
        "nl2sql_vector_store_base_dir": config.get("nl2sql_vector_store_base_dir"),
        "llm_provider": active_profile.get("provider"),
        # Correctly format the model_id for litellm
        "model_id": f"{active_profile.get('provider')}/{active_profile.get('model_id')}",
        # Add database connection info
        "active_database_alias": config.get("active_database_alias"),
        "active_database_connection_string": config.get("database_connections", {}).get(config.get("active_database_alias"))
    }
    
    # Add credentials to the top level of the app_config
    credentials = active_profile.get("credentials", {})
    app_config.update(credentials)

    # Set environment variables for LiteLLM from the active profile
    # _set_env_vars_from_profile(active_profile) # Temporarily disabled for debugging

    return app_config

if __name__ == '__main__':
    # Example usage:
    config = get_app_config()
    print("\nCurrent Application Configuration:")
    print(json.dumps(config, indent=4))

    # Test specific paths
    if config:
        print(f"\nMemory base directory (insights, schema, history, vector stores): {config.get('memory_base_dir')}")
        print(f"Approved Queries (NLQ-SQL JSON pairs) directory: {config.get('approved_queries_dir')}")
        print(f"LanceDB vector store base directory: {config.get('nl2sql_vector_store_base_dir')}")
        print(f"LLM Provider: {config.get('llm_provider')}")
        print(f"LLM Model: {config.get('model_id')}")
