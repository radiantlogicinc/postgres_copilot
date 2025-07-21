import asyncio
import asyncio
import os
import sys
import subprocess # Keep for StdioServerParameters if server is run as subprocess
# from dotenv import load_dotenv # Will be handled by config_manager
from typing import Optional, Any, Dict, Tuple
from pathlib import Path # Added
from contextlib import AsyncExitStack
import datetime
import litellm
from pydantic import ValidationError # For catching Pydantic errors
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool as McpTool 
import copy
import re
import json
import memory_module
import initialization_module
import sql_generation_module
import insights_module
import revision_insights_module # Added
import database_navigation_module
import revise_query_module # Added
import vector_store_module # Added for RAG
import model_change_module # Added for /change_model
import error_handler_module
import token_utils
from token_logging_module import log_token_usage
from colorama import Fore, Style, init as colorama_init # Added for Colorama
import config_manager # Changed to absolute import
import inspect
from pydantic_models import ( # Changed to absolute import
    SQLGenerationResponse, 
    FeedbackReportContentModel, 
    FeedbackIteration,
    RevisionReportContentModel, # Added
    RevisionIteration # Added
)

# --- Configuration Setup ---
# Configuration, including API keys and paths, is now handled by config_manager.py
# The get_app_config() call in main() will ensure this is loaded/set up.

# LiteLLM doesn't require a global configure call.
# API keys are set as environment variables by config_manager.py based on user's choice.
# LiteLLM will automatically use the appropriate credentials based on the model ID prefix.

class LiteLLMMcpClient:
    """A client that connects to an MCP server and uses LiteLLM for interaction."""

    def __init__(self, app_config: dict, system_instruction: Optional[str] = None): # Added app_config
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.app_config = app_config # Store app_config

        self.model_name = self.app_config.get("model_id") # Get from app_config
        self.provider = self.app_config.get("llm_provider")
        self.llm_api_key = self.app_config.get("api_key") # Store for potential direct use if needed
        
        # Determine model provider based on model name prefix
        # Provider name from config_data['llm_provider'] can also be used directly
        self.model_provider = self.app_config.get("llm_provider", "Unknown").capitalize()
        # Fallback logic if llm_provider not in config, or to be more specific
        if self.model_name.startswith("gemini/"):
            self.model_provider = "Google AI (Gemini)"
        elif self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
            self.model_provider = "OpenAI"
        elif self.model_name.startswith("bedrock/"):
            self.model_provider = "AWS Bedrock"
            if "claude" in self.model_name.lower():
                self.model_provider += " (Anthropic Claude)"
        elif self.model_name.startswith("anthropic/") or self.model_name.startswith("claude"):
            self.model_provider = "Anthropic"
        elif self.model_name.startswith("ollama/"):
            self.model_provider = "Ollama (Local)"
        
        self.system_instruction_content = system_instruction
        self.conversation_history: list[Dict[str, Any]] = []
        if self.system_instruction_content:
            self.conversation_history.append({"role": "system", "content": self.system_instruction_content})

        # RAG and Display Thresholds are now hardcoded in vector_store_module.
        # Removing client-level attributes for these.
        # self.rag_similarity_threshold = vector_store_config.LITELLM_RAG_THRESHOLD (Removed)
        # self.display_similarity_threshold = vector_store_config.LITELLM_DISPLAY_THRESHOLD (Removed)
        
        # Memory directories are ensured by memory_module on its import.
        # No need for verbose printing of these paths here.

        # Core state variables
        self.is_initialized: bool = False
        self.current_db_connection_string: Optional[str] = None
        self.current_db_name_identifier: Optional[str] = None # e.g., "california_schools_db"
        
        # Data used for context
        self.db_schema_and_sample_data: Optional[Dict[str, Any]] = None # Loaded from memory/schema/
        self.cumulative_insights_content: Optional[str] = None # Loaded from memory/insights/summarized_insights.md

        # State for the current query/feedback cycle
        self.current_natural_language_question: Optional[str] = None
        # Holds the Pydantic model of the full feedback report being built/iterated on
        self.current_feedback_report_content: Optional[FeedbackReportContentModel] = None
        # Note: last_generated_sql, last_sql_explanation etc. are now part of current_feedback_report_content
        
        # State for query revision feature
        self.current_revision_report_content: Optional[RevisionReportContentModel] = None
        self.is_in_revision_mode: bool = False
        self.feedback_used_in_current_revision_cycle: bool = False
        self.feedback_log_in_revision: list[Dict[str, str]] = []
        self.active_table_scope: Optional[List[str]] = None
        self.table_categories: Optional[Dict[str, List[str]]] = None

    # The first_run concept is now implicitly handled by config_manager.get_app_config()
    # which triggers initial_setup() if config.json doesn't exist.
    # We can remove _get_first_run_flag_path and _check_and_set_first_run.
    # The welcome message can be shown if initial_setup was triggered.

    async def _cleanup_database_session(self, full_cleanup: bool = True):
        # print("Cleaning up database session state...") # User doesn't need to see this
        if full_cleanup:
            self.is_initialized = False
            self.current_db_connection_string = None
            self.current_db_name_identifier = None
            self.db_schema_and_sample_data = None
            self.cumulative_insights_content = None 
        
        self._reset_feedback_cycle_state()
        self._reset_revision_cycle_state() # Added
        # print("Database session state cleaned.")

    def _reset_feedback_cycle_state(self):
        """Resets state for a new natural language query and its feedback cycle."""
        # print("Resetting feedback cycle state for new SQL generation...") # Internal detail
        self.current_natural_language_question = None
        self.current_feedback_report_content = None
        # If starting a new feedback cycle, revision mode should also reset
        self._reset_revision_cycle_state()


    def _reset_revision_cycle_state(self):
        """Resets state for a new query revision cycle."""
        # print("Resetting revision cycle state...") # Internal detail
        self.current_revision_report_content = None
        self.is_in_revision_mode = False
        self.feedback_used_in_current_revision_cycle = False
        self.feedback_log_in_revision = []

    def _extract_connection_string_and_db_name(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        # Extracts connection string and attempts to derive a db_name from it.
        # Example: postgresql://user:password@host:port/dbname
        conn_str_match = re.search(r"(postgresql://\S+:\S+@\S+:\d+/(\S+))", query)
        if conn_str_match:
            full_conn_str = conn_str_match.group(1)
            db_name_part = conn_str_match.group(2)
            # Sanitize db_name_part to be a valid filename component
            sanitized_db_name = re.sub(r'[^\w\-_\.]', '_', db_name_part) if db_name_part else "unknown_db"
            return full_conn_str, sanitized_db_name
        return None, None

    def _extract_mcp_tool_call_output(self, tool_call_result: Any, tool_name: str) -> Any:
        if not self.session: return "Error: MCP session not available."
        
        output = None
        if hasattr(tool_call_result, 'content') and isinstance(tool_call_result.content, list) and \
           len(tool_call_result.content) > 0 and hasattr(tool_call_result.content[0], 'text') and \
           tool_call_result.content[0].text is not None:
            output = tool_call_result.content[0].text
        elif hasattr(tool_call_result, 'output'): output = tool_call_result.output
        elif hasattr(tool_call_result, 'result'): output = tool_call_result.result
        elif tool_call_result is not None:
            if not isinstance(tool_call_result, (str, int, float, bool, list, dict)):
                try: output = str(tool_call_result)
                except Exception as e: output = f"Error: Tool output format not recognized and failed to convert to string. Type: {type(tool_call_result)}, Error: {e}"
            else: output = tool_call_result
        else:
            output = "Error: Tool output not found or format not recognized."

        # Log the raw response
        error_handler_module.log_mcp_response(tool_name, output)
        return output

    async def connect_to_mcp_server(self, server_script_path: Path): # Changed type hint to Path
        # print(f"Connecting to MCP server script: {server_script_path}") # Internal detail
        
        # server_script_path is now expected to be an absolute Path object from main()
        is_python = server_script_path.suffix == '.py' # Use Path.suffix
        command = sys.executable # sys.executable is a string path to the python interpreter
        
        # Ensure args for StdioServerParameters are strings
        server_args = [str(server_script_path)]
        
        # The command to run is the python interpreter if it's a .py script
        # If it were a node script, command would be 'node', etc.
        # StdioServerParameters takes the command and then its arguments.
        # So, if is_python, command is sys.executable, and server_args is [path_to_script.py]
        
        server_params = StdioServerParameters(command=command, args=server_args, env=None)
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read_stream, write_stream = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await self.session.initialize()
            response = await self.session.list_tools()
            mcp_tools_list: list[McpTool] = response.tools
            # print(f"Connected to MCP server with {len(mcp_tools_list)} tools: {[tool.name for tool in mcp_tools_list]}") # Internal detail
            if not mcp_tools_list:
                error_handler_module.display_message("No tools discovered from the MCP server.", level="FATAL")

            self.litellm_tools = []
            for mcp_tool_obj in mcp_tools_list:
                tool_name = getattr(mcp_tool_obj, 'name', None)
                tool_desc = getattr(mcp_tool_obj, 'description', None)
                tool_schema = copy.deepcopy(getattr(mcp_tool_obj, 'inputSchema', {}))
                
                if tool_schema:
                    if 'properties' not in tool_schema and tool_schema:
                        pass # No debug print
                    elif 'properties' in tool_schema:
                         tool_schema['type'] = 'object'
                    tool_schema.pop('title', None)
                    if 'properties' in tool_schema and isinstance(tool_schema['properties'], dict):
                        for prop_name in list(tool_schema['properties'].keys()):
                            if isinstance(tool_schema['properties'][prop_name], dict):
                                tool_schema['properties'][prop_name].pop('title', None)
                
                if tool_name and tool_desc:
                    self.litellm_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_desc,
                            "parameters": tool_schema or {"type": "object", "properties": {}}
                        }
                    })
                else:
                    error_handler_module.display_message(f"Skipping MCP tool '{tool_name}' due to missing attributes for LiteLLM conversion.", level="WARNING")
            
            if not self.litellm_tools and mcp_tools_list:
                error_handler_module.display_message("No MCP tools could be converted to LiteLLM format.", level="FATAL")

            # print(f"LiteLLM client configured to use model '{self.model_name}' with {len(self.litellm_tools)} tools.") # Internal detail
        except Exception as e:
            error_handler_module.display_message(f"Fatal error during MCP server connection or LiteLLM setup: {e}. Please ensure the MCP server script is correct and executable.", level="FATAL")
            await self.cleanup()


    async def _send_message_to_llm(self, messages: list, user_query: str, schema_tokens: int = 0, tools: Optional[list] = None, tool_choice: str = "auto",
                                  response_format: Optional[dict] = None) -> Any:
        """Sends messages to LiteLLM and handles response, including tool calls."""
        try:
            # Get caller info
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            origin_script = caller_frame.f_code.co_filename
            origin_line = caller_frame.f_lineno

            # Prepare kwargs for acompletion
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }

            # Add provider-specific credentials from app_config
            provider = self.app_config.get("llm_provider")
            if provider == "bedrock":
                kwargs["aws_access_key_id"] = self.app_config.get("access_key_id")
                kwargs["aws_secret_access_key"] = self.app_config.get("secret_access_key")
                kwargs["aws_region_name"] = self.app_config.get("region")
            else:
                # Default to using api_key for other providers
                kwargs["api_key"] = self.app_config.get("api_key")
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice
            
            # Add response_format if provided (for JSON responses with OpenAI models)
            if response_format and (self.model_name.startswith("gpt-") or self.model_name.startswith("openai/")):
                kwargs["response_format"] = response_format
            
            try:
                response = await litellm.acompletion(**kwargs)
                # Log token usage on success
                prompt_text = messages[-1]['content']
                prompt_tokens = token_utils.count_tokens(prompt_text, self.model_name, self.provider)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                llm_response = response.choices[0].message.content

                log_token_usage(
                    origin_script=origin_script,
                    origin_line=origin_line,
                    user_query=user_query,
                    prompt=prompt_text,
                    prompt_tokens=prompt_tokens,
                    schema_tokens=schema_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    llm_response=llm_response,
                    model_id=self.model_name
                )
                
                # Add user message to history (assistant response added after processing)
                # The last message in `messages` is the current user prompt
                if messages[-1]["role"] == "user":
                     self.conversation_history.append(messages[-1])

                return response
            except litellm.RateLimitError as e:
                # Log token usage on rate limit error
                prompt_text = messages[-1]['content']
                prompt_tokens = token_utils.count_tokens(prompt_text, self.model_name, self.provider)
                log_token_usage(
                    origin_script=origin_script,
                    origin_line=origin_line,
                    user_query=user_query,
                    prompt=prompt_text,
                    prompt_tokens=prompt_tokens,
                    schema_tokens=schema_tokens,
                    input_tokens=prompt_tokens, # Input tokens are the prompt tokens
                    output_tokens=0, # No output tokens
                    llm_response=f"RateLimitError: {e}",
                    model_id=self.model_name
                )
                raise e # Re-raise the exception to be handled by the caller
        except Exception as e:
            # Do not log here for other exceptions; the caller is responsible for handling and logging.
            # Just add a placeholder to history and re-raise.
            if messages[-1]["role"] == "user" and messages[-1] not in self.conversation_history:
                 self.conversation_history.append(messages[-1])
            self.conversation_history.append({"role": "assistant", "content": "I'm having trouble processing your request."})
            raise # Re-raise the exception to be handled by the caller

    async def _process_llm_response(self, llm_response: Any) -> Tuple[str, bool]:
        """Processes LiteLLM response, handles tool calls, and returns text response and if a tool was called."""
        assistant_response_content = ""
        tool_calls_made = False

        if not llm_response or not llm_response.choices or not llm_response.choices[0].message:
            error_handler_module.display_message("Empty or invalid response from LLM.", level="ERROR")
            assistant_response_content = "Error: Empty or invalid response from LLM." # Keep for history
            self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
            return assistant_response_content, tool_calls_made

        message = llm_response.choices[0].message
        
        # Storing the raw assistant message (potentially with tool_calls)
        assistant_message_for_history = {"role": "assistant"}
        if message.content:
            assistant_message_for_history["content"] = message.content
            assistant_response_content = message.content # Initial text part

        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls_made = True
            assistant_message_for_history["tool_calls"] = [] # For history
            
            # LiteLLM returns tool_calls in OpenAI format
            # [{"id": "call_abc", "type": "function", "function": {"name": "tool_name", "arguments": "{...}"}}]
            
            tool_call_responses_for_next_llm_call = []

            for tool_call in message.tool_calls:
                tool_call_id = tool_call.id
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                
                # Add the tool call itself to history
                assistant_message_for_history["tool_calls"].append({
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args_str}
                })

                try:
                    tool_args = json.loads(tool_args_str)
                    mcp_tool_result_obj = await self.session.call_tool(tool_name, tool_args)
                    tool_output = self._extract_mcp_tool_call_output(mcp_tool_result_obj, tool_name)
                except json.JSONDecodeError as e_json:
                    tool_output = f"Error: Invalid JSON arguments for tool {tool_name}: {e_json}. Arguments received: {tool_args_str}"
                    error_handler_module.display_message(f"MCP Tool Error: Invalid JSON arguments for tool {tool_name}: {e_json}. Args: {tool_args_str}", level="ERROR")
                except Exception as e_tool:
                    tool_output = f"Error executing tool {tool_name}: {e_tool}"
                    error_handler_module.display_message(f"Error executing tool {tool_name}: {e_tool}", level="ERROR")

                tool_call_responses_for_next_llm_call.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": tool_name,
                    "content": str(tool_output) # Ensure content is string
                })

            # Add the assistant's message (that included the tool call request) to history
            self.conversation_history.append(assistant_message_for_history)

            # Add all tool responses to history
            for resp in tool_call_responses_for_next_llm_call:
                self.conversation_history.append(resp) 

            # Make a follow-up call to LLM with tool responses
            # print("Sending tool responses back to LLM...") # Debug
            
            # The conversation history already contains:
            # ..., user_prompt, assistant_tool_call_request, tool_response_1, ...
            # So we can just send the current self.conversation_history
            
            follow_up_llm_response = await self._send_message_to_llm(self.conversation_history, user_query, tools=self.litellm_tools)
            
            # Process this new response (which should be the final text from LLM after tools)
            # This recursive call is safe as long as LLM doesn't loop infinitely on tool calls
            assistant_response_content, _ = await self._process_llm_response(follow_up_llm_response)
            # The _process_llm_response will handle adding the final assistant message to history.

        else: # No tool calls, just a direct text response
            if message.content: # Ensure there's content
                assistant_response_content = message.content
                # Add assistant's direct response to history
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})
            else: # Should not happen if the first check passed, but as a safeguard
                error_handler_module.display_message("LLM response message has no content.", level="ERROR")
                assistant_response_content = "Error: LLM response message has no content." # Keep for history
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})


        return assistant_response_content, tool_calls_made


    async def _handle_initialization(self, connection_string: str, db_name_id: str):
        """Handles the full DB initialization flow."""
        await self._cleanup_database_session(full_cleanup=True) # Clean slate for new DB
        
        success, message, schema_data = await initialization_module.perform_initialization(
            self, connection_string, db_name_id
        )
        if success:
            self.is_initialized = True
            self.current_db_connection_string = connection_string
            self.current_db_name_identifier = db_name_id
            self.db_schema_and_sample_data = schema_data
            # Load cumulative insights for this specific DB if they exist
            self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
            self._reset_feedback_cycle_state() # Ensure clean state for new DB
            self._reset_revision_cycle_state() # Ensure clean state for new DB
            error_handler_module.display_response(message)
        else:
            await self._cleanup_database_session(full_cleanup=True) 
            error_handler_module.display_message(message, level="ERROR")

    async def dispatch_command(self, query: str):
        if not self.session:
            error_handler_module.display_message("Client not fully initialized (MCP session missing).", level="ERROR")
            return

        # Flexible command parsing: command can be followed by ':' or space, then argument
        command_match = re.match(r"/(\w+)(?:\s*:?\s*)(.*)", query, re.IGNORECASE)
        
        base_command_lower = ""
        argument_text = ""

        if command_match:
            base_command_lower = command_match.group(1).lower()
            argument_text = command_match.group(2).strip()
        elif query.startswith("/"): # Handles commands like /approved with no args
            base_command_lower = query[1:].lower()
        # If not a command starting with "/", it will be handled by implicit init or navigation query later

        # Command: /change_model
        if base_command_lower == "change_model":
            # The new handle_change_model_interactive saves the config file itself.
            # We need to reload the config if it returns success.
            success, message = await model_change_module.handle_change_model_interactive(self.app_config)
            
            if success:
                print("Reloading configuration with new profile...")
                # Reload the app_config from the updated file
                self.app_config = config_manager.get_app_config()
                
                # Update client's internal LLM settings from the newly loaded app_config
                self.model_name = self.app_config.get("model_id")
                self.llm_api_key = self.app_config.get("api_key")
                
                new_provider_name = self.app_config.get("llm_provider", "Unknown").capitalize()
                # This logic correctly re-evaluates the provider display name
                if self.model_name.startswith("gemini/"): self.model_provider = "Google AI (Gemini)"
                elif self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"): self.model_provider = "OpenAI"
                elif self.model_name.startswith("bedrock/"):
                    self.model_provider = "AWS Bedrock"
                    if "claude" in self.model_name.lower(): self.model_provider += " (Anthropic Claude)"
                elif self.model_name.startswith("anthropic/") or self.model_name.startswith("claude"): self.model_provider = "Anthropic"
                elif self.model_name.startswith("ollama/"): self.model_provider = "Ollama (Local)"
                else: self.model_provider = new_provider_name
                
                # Reset conversation history as the context might be irrelevant for a new model/provider
                self.conversation_history = []
                if self.system_instruction_content:
                   self.conversation_history.append({"role": "system", "content": self.system_instruction_content})
                
                final_message = (
                    f"{message}\n"
                    f"Client updated to use new profile. Active model: {self.model_name} from {self.model_provider}.\n"
                    f"Conversation history has been reset."
                )
                error_handler_module.display_response(final_message)
            else:
                # Display cancellation or error message from the module
                error_handler_module.display_response(message)
            return

        # Command: /change_profile
        if base_command_lower == "change_profile":
            config = config_manager.load_config()
            profiles = config.get("llm_profiles", {})
            if len(profiles) <= 1:
                error_handler_module.display_response("Only one profile exists. Add more profiles to use this command.")
                return

            print("Please choose a profile to switch to:")
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
            
            chosen_alias = profile_aliases[choice - 1]
            config["active_llm_profile_alias"] = chosen_alias
            config_manager.save_config(config)
            
            print(f"Reloading configuration with new profile '{chosen_alias}'...")
            # Manually rebuild app_config to avoid the double prompt from get_app_config()
            active_profile = config.get("llm_profiles", {}).get(chosen_alias)
            if not active_profile:
                error_handler_module.display_message(f"Error: Could not find the selected profile '{chosen_alias}' after saving.", level="ERROR")
                return

            app_config = {
                "memory_base_dir": config.get("memory_base_dir"),
                "approved_queries_dir": config.get("approved_queries_dir"),
                "nl2sql_vector_store_base_dir": config.get("nl2sql_vector_store_base_dir"),
                "llm_provider": active_profile.get("provider"),
                "model_id": f"{active_profile.get('provider')}/{active_profile.get('model_id')}",
                "active_database_alias": config.get("active_database_alias"),
                "active_database_connection_string": config.get("database_connections", {}).get(config.get("active_database_alias"))
            }
            credentials = active_profile.get("credentials", {})
            app_config.update(credentials)
            self.app_config = app_config

            self.model_name = self.app_config.get("model_id")
            self.llm_api_key = self.app_config.get("api_key")
            new_provider_name = self.app_config.get("llm_provider", "Unknown").capitalize()
            if self.model_name.startswith("gemini/"): self.model_provider = "Google AI (Gemini)"
            elif self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"): self.model_provider = "OpenAI"
            elif self.model_name.startswith("bedrock/"):
                self.model_provider = "AWS Bedrock"
                if "claude" in self.model_name.lower(): self.model_provider += " (Anthropic Claude)"
            elif self.model_name.startswith("anthropic/") or self.model_name.startswith("claude"): self.model_provider = "Anthropic"
            elif self.model_name.startswith("ollama/"): self.model_provider = "Ollama (Local)"
            else: self.model_provider = new_provider_name
            self.conversation_history = []
            if self.system_instruction_content:
                self.conversation_history.append({"role": "system", "content": self.system_instruction_content})
            final_message = (
                f"Active profile switched to '{chosen_alias}'.\n"
                f"Client updated to use new profile. Active model: {self.model_name} from {self.model_provider}.\n"
                f"Conversation history has been reset."
            )
            error_handler_module.display_response(final_message)
            return

        # Handle initialization attempts first
        elif base_command_lower == "change_database":
            raw_conn_str = argument_text
            if raw_conn_str.startswith('"') and raw_conn_str.endswith('"'): # Handle quoted string
                raw_conn_str = raw_conn_str[1:-1]
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(raw_conn_str)
            if not parsed_conn_str:
                error_handler_module.display_message("Invalid connection string format provided with /change_database. Expected: postgresql://user:pass@host:port/dbname", level="ERROR")
                return
            await self._handle_initialization(parsed_conn_str, parsed_db_name_id)

        # Check for implicit initialization if not already initialized
        elif not self.is_initialized:
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(query) # Use original query for implicit check
            if parsed_conn_str:
                await self._handle_initialization(parsed_conn_str, parsed_db_name_id)
                return # Exit after handling implicit initialization
            else:
                # If not initialized and not an attempt to initialize (via /change_database or raw string),
                # prompt specifically for connection.
                error_handler_module.display_message("Database not initialized. Please provide a connection string (e.g., postgresql://user:pass@host:port/dbname) or use '/change_database {connection_string}' to connect.", level="ERROR")
                return # Exit after showing the error

        # If we reach here, self.is_initialized must be True.
        # Perform a redundant check just in case, though the logic above should ensure it.
        elif not self.is_initialized or not self.current_db_name_identifier:
            # This state should ideally not be reached if the above logic is correct.
            error_handler_module.display_message("Critical Error: Database initialization state is inconsistent. Please try /change_database again or provide a connection string.", level="ERROR")
            return

        # Command: /reload_scope
        elif base_command_lower == "reload_scope":
            if not self.is_initialized or not self.current_db_name_identifier:
                error_handler_module.display_message("Please connect to a database first with /change_database.", level="ERROR")
                return
            
            print("Reloading database scope...")
            filtered_schema, message = initialization_module.load_and_filter_schema(self.current_db_name_identifier)
            
            if filtered_schema is not None:
                self.db_schema_and_sample_data = filtered_schema
                error_handler_module.display_response(message)
            else:
                error_handler_module.display_message(f"Failed to reload scope: {message}", level="ERROR")
            return

        # Command: /generate_sql
        elif base_command_lower == "generate_sql":
            nl_question = argument_text
            if not nl_question:
                error_handler_module.display_message("Please provide a natural language question after /generate_sql.", level="ERROR")
                return
            self._reset_feedback_cycle_state()
            self._reset_revision_cycle_state()
            self.current_natural_language_question = nl_question
            print("Generating SQL, please wait...")
            sql_gen_result_dict = await sql_generation_module.generate_sql_query(
                self, nl_question, self.db_schema_and_sample_data, self.cumulative_insights_content,
                row_limit_for_preview=1 # Ensure 1 row for preview from sql_generation_module
            )
            
            if sql_gen_result_dict.get("sql_query"):
                self.current_feedback_report_content = FeedbackReportContentModel(
                    natural_language_question=nl_question,
                    initial_sql_query=sql_gen_result_dict["sql_query"],
                    initial_explanation=sql_gen_result_dict.get("explanation", "N/A"),
                    final_corrected_sql_query=sql_gen_result_dict["sql_query"], 
                    final_explanation=sql_gen_result_dict.get("explanation", "N/A") 
                )
            else: 
                 self.current_feedback_report_content = None 
            
            base_message_to_user = sql_gen_result_dict.get("message_to_user", "Error: No message from SQL generation.")
            
            # Append execution result or error to the message_to_user
            exec_result = sql_gen_result_dict.get("execution_result")
            exec_error = sql_gen_result_dict.get("execution_error")

            # If there's an execution error from the generation loop (like ContextWindowExceeded),
            # it should be displayed directly, and we shouldn't proceed with other formatting.
            if exec_error and not sql_gen_result_dict.get("sql_query"):
                error_handler_module.display_message(exec_error, level="ERROR")
                return

            if exec_error:
                base_message_to_user += f"\nExecution Error: {exec_error}\n"
            elif exec_result is not None:
                preview_str = ""
                if isinstance(exec_result, dict) and exec_result.get("status") == "success":
                    data = exec_result.get("data")
                    if data and isinstance(data, list) and len(data) > 0:
                        preview_str = str(data[0])
                    elif 'message' in exec_result:
                        preview_str = exec_result['message']
                    else:
                        preview_str = "Query executed successfully, but no rows were returned."
                else:
                    preview_str = str(exec_result)

                if len(preview_str) > 200: # Truncate if too long
                    preview_str = preview_str[:197] + "..."
                base_message_to_user += f"\nExecution successful. Result preview (1 row): {preview_str}\n"
            
            # --- Augment message with display few-shot examples ---
            # Ensure base_message_to_user is a string before appending.
            if isinstance(base_message_to_user, str):
                if self.current_db_name_identifier and self.current_natural_language_question:
                    try:
                        # Using hardcoded display threshold from vector_store_module
                        display_threshold_from_module = vector_store_module.LITELLM_DISPLAY_THRESHOLD
                        display_examples = vector_store_module.search_similar_nlqs(
                            db_name_identifier=self.current_db_name_identifier,
                            query_nlq=self.current_natural_language_question,
                            k=3, # Show up to 3 examples
                            threshold=display_threshold_from_module
                        )
                        if display_examples:
                            display_message_parts = ["\n\n--- Relevant Approved Examples (Similarity >= " f"{display_threshold_from_module}" ") ---"]
                            for i, ex in enumerate(display_examples):
                                display_message_parts.append(f"Example {i+1} (Similarity: {ex['similarity_score']:.2f}):")
                                display_message_parts.append(f"  Q: \"{ex['nlq']}\"")
                                display_message_parts.append(f"  A: ```sql\n{ex['sql']}\n```")
                            base_message_to_user += "\n" + "\n".join(display_message_parts)
                        else:
                            # This is not an error, just informational.
                            base_message_to_user += f"\n\n--- No similar approved examples found (Similarity >= {display_threshold_from_module}) ---"
                    except Exception as e_display_rag:
                        # Log the exception but don't crash the main flow.
                        error_handler_module.handle_exception(e_display_rag, self.current_natural_language_question, {"context": "Display RAG examples"})
                        base_message_to_user += "\n\n--- Could not retrieve similar examples due to an error. ---"
            # --- End Augment ---
            
            error_handler_module.display_response(base_message_to_user)

        # Command: /feedback
        elif base_command_lower == "feedback":
            user_feedback_text = argument_text
            if not user_feedback_text:
                error_handler_module.display_message("Please provide your feedback text after /feedback.", level="ERROR")
                return
            print("Processing feedback, please wait...")

            if self.is_in_revision_mode and self.current_revision_report_content and self.current_revision_report_content.final_revised_sql_query:
                # Apply feedback to the current revision
                current_sql = self.current_revision_report_content.final_revised_sql_query
                current_explanation = self.current_revision_report_content.final_revised_explanation or "N/A"
                
                # Prompt for correcting SQL based on feedback (within revision context)
                feedback_prompt_for_revision = (
                    f"You are an expert PostgreSQL SQL assistant. A user is providing feedback on a previously revised SQL query.\n"
                    f"CURRENT REVISED SQL QUERY:\n```sql\n{current_sql}\n```\n"
                    f"ITS EXPLANATION:\n{current_explanation}\n\n"
                    f"USER FEEDBACK: \"{user_feedback_text}\"\n\n"
                    f"Based on this feedback, please provide a corrected SQL query (must start with SELECT) and a brief explanation for the correction.\n"
                    f"Respond ONLY with a single JSON object matching this structure: "
                    f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation for the correction>\" }}\n"
                )
                MAX_FEEDBACK_RETRIES_IN_REVISION = 1
                corrected_sql_from_feedback = None
                corrected_explanation_from_feedback = None

                for attempt in range(MAX_FEEDBACK_RETRIES_IN_REVISION + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt_for_revision}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        llm_response_obj = await self._send_message_to_llm(messages=messages_for_llm, user_query=user_feedback_text, response_format=response_format)
                        response_text, _ = await self._process_llm_response(llm_response_obj)

                        if response_text.startswith("```json"): response_text = response_text[7:]
                        if response_text.endswith("```"): response_text = response_text[:-3]
                        
                        parsed_correction = SQLGenerationResponse.model_validate_json(response_text.strip())
                        
                        if not parsed_correction.sql_query or not parsed_correction.sql_query.strip().upper().startswith("SELECT"):
                            raise ValueError("Corrected SQL from feedback must start with SELECT.")
                        
                        corrected_sql_from_feedback = parsed_correction.sql_query
                        corrected_sql_from_feedback = parsed_correction.sql_query
                        corrected_explanation_from_feedback = parsed_correction.explanation or "N/A"

                        # Log feedback for potential report generation on approve_revision
                        self.feedback_used_in_current_revision_cycle = True
                        self.feedback_log_in_revision.append({
                            "user_feedback_text": user_feedback_text,
                            "sql_before_feedback": current_sql, # SQL before this feedback
                            "explanation_before_feedback": current_explanation, # Explanation for current_sql
                            "corrected_sql_attempt": corrected_sql_from_feedback, # SQL after this feedback
                            "corrected_explanation": corrected_explanation_from_feedback # Explanation for new SQL
                        })
                        break 
                    except (ValidationError, json.JSONDecodeError, ValueError) as e:
                        if attempt == MAX_FEEDBACK_RETRIES_IN_REVISION:
                            error_handler_module.display_message(f"Error processing feedback on revised query: {e}. Please try rephrasing your feedback.", level="ERROR")
                            return
                        feedback_prompt_for_revision = f"Your previous attempt to correct the SQL based on feedback was invalid (Error: {e}). Please try again, ensuring the JSON output has 'sql_query' (starting with SELECT) and 'explanation'."
                    except Exception as e_gen:
                        if attempt == MAX_FEEDBACK_RETRIES_IN_REVISION:
                            error_handler_module.display_message(f"Unexpected error processing feedback on revised query: {e_gen}.", level="ERROR")
                            return
                        feedback_prompt_for_revision = "An unexpected error occurred. Please try to regenerate the corrected SQL and explanation based on the feedback."
                
                if corrected_sql_from_feedback and self.current_revision_report_content:
                    new_iteration = RevisionIteration(
                        user_revision_prompt=f"Feedback: {user_feedback_text}", 
                        revised_sql_attempt=corrected_sql_from_feedback,
                        revised_explanation=corrected_explanation_from_feedback
                    )
                    self.current_revision_report_content.revision_iterations.append(new_iteration)
                    self.current_revision_report_content.final_revised_sql_query = corrected_sql_from_feedback
                    self.current_revision_report_content.final_revised_explanation = corrected_explanation_from_feedback

                    exec_result, exec_error = None, None
                    try:
                        exec_obj = await self.session.call_tool("execute_postgres_query", {"query": corrected_sql_from_feedback, "row_limit": 1})
                        raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                        if isinstance(raw_output, str) and "Error:" in raw_output:
                            exec_error = raw_output
                        else:
                            exec_result = raw_output
                    except Exception as e_exec:
                        exec_error = str(e_exec)

                    user_msg = f"Feedback applied to revised query. New SQL attempt:\n```sql\n{corrected_sql_from_feedback}\n```\n"
                    user_msg += f"Explanation:\n{corrected_explanation_from_feedback}\n"
                    if exec_error: 
                        user_msg += f"\nExecution Error for new SQL: {exec_error}\n"
                    elif exec_result is not None:
                        preview_str = ""
                        if isinstance(exec_result, list) and len(exec_result) == 1 and isinstance(exec_result[0], dict):
                            single_row_dict = exec_result[0]
                            preview_str = str(single_row_dict)
                        elif isinstance(exec_result, str) and exec_result.endswith(".md"):
                             preview_str = f"Query result saved to {os.path.basename(exec_result)}"
                        else:
                            preview_str = str(exec_result)

                        if len(preview_str) > 200:
                            preview_str = preview_str[:197] + "..."
                        user_msg += f"\nExecution of new SQL successful. Result preview (1 row): {preview_str}\n"
                    user_msg += "Use `/revise Your new prompt`, more `/feedback`, or `/approve_revision`."
                    error_handler_module.display_response(user_msg)
                else:
                    error_handler_module.display_message("Failed to apply feedback to the revised query.", level="ERROR")

            elif self.current_feedback_report_content and self.current_natural_language_question:
                self._reset_revision_cycle_state() 
                current_report_json = self.current_feedback_report_content.model_dump_json(indent=2)
                feedback_model_schema_dict = FeedbackReportContentModel.model_json_schema()
                feedback_model_schema_str = json.dumps(feedback_model_schema_dict, indent=2)

                feedback_prompt = (
                    f"You are refining a SQL query based on user feedback and updating a detailed report.\n"
                    f"The user's original question was: \"{self.current_natural_language_question}\"\n"
                    f"The current state of the feedback report (JSON format) is:\n```json\n{current_report_json}\n```\n"
                    f"The user has provided new feedback: \"{user_feedback_text}\"\n\n"
                    f"Your tasks:\n"
                    f"1. Generate a new `corrected_sql_attempt` and `corrected_explanation` based on this latest feedback and the *previous* `final_corrected_sql_query` from the report.\n"
                    f"2. Create a new `FeedbackIteration` object containing this `user_feedback_text`, your new `corrected_sql_attempt`, and `corrected_explanation`.\n"
                    f"3. Append this new `FeedbackIteration` to the `feedback_iterations` list in the report.\n"
                    f"4. Update the report's `final_corrected_sql_query` and `final_explanation` to your latest attempt.\n"
                    f"5. Re-evaluate and update the LLM analysis sections: `why_initial_query_was_wrong_or_suboptimal` (if applicable, comparing to initial), `why_final_query_works_or_is_improved` (explaining your latest correction), `database_insights_learned_from_this_query`, and `sql_lessons_learned_from_this_query` based on the *entire* history including this new iteration.\n\n"
                    f"Respond ONLY with the complete, updated JSON object for the `FeedbackReportContentModel`, conforming to this schema:\n"
                    f"```json\n{feedback_model_schema_str}\n```\n"
                    f"Ensure the `corrected_sql_attempt` and `final_corrected_sql_query` start with SELECT."
                )
                
                MAX_FEEDBACK_RETRIES = 1
                for attempt in range(MAX_FEEDBACK_RETRIES + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        llm_response_obj = await self._send_message_to_llm(
                            messages=messages_for_llm,
                            user_query=user_feedback_text,
                            response_format=response_format
                        )
                        response_text, _ = await self._process_llm_response(llm_response_obj)

                        if response_text.startswith("```json"): response_text = response_text[7:]
                        if response_text.endswith("```"): response_text = response_text[:-3]
                        
                        updated_report_model = FeedbackReportContentModel.model_validate_json(response_text.strip())
                        
                        if not updated_report_model.final_corrected_sql_query or \
                           not updated_report_model.final_corrected_sql_query.strip().upper().startswith("SELECT"):
                            raise ValueError("Corrected SQL in feedback report must start with SELECT.")

                        self.current_feedback_report_content = updated_report_model
                        
                        exec_result, exec_error = None, None
                        try:
                            exec_obj = await self.session.call_tool("execute_postgres_query", {"query": updated_report_model.final_corrected_sql_query, "row_limit": 1})
                            raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                            if isinstance(raw_output, str) and "Error:" in raw_output:
                                exec_error = raw_output
                            else:
                                exec_result = raw_output
                        except Exception as e_exec:
                            exec_error = str(e_exec)

                        user_msg = f"Feedback processed. New SQL attempt:\n```sql\n{updated_report_model.final_corrected_sql_query}\n```\n"
                        user_msg += f"Explanation:\n{updated_report_model.final_explanation}\n"
                        if exec_error:
                            user_msg += f"\nExecution Error for new SQL: {exec_error}\n"
                        elif exec_result is not None:
                            preview_str = ""
                            if isinstance(exec_result, list) and len(exec_result) == 1 and isinstance(exec_result[0], dict):
                                single_row_dict = exec_result[0]
                                preview_str = str(single_row_dict)
                            elif isinstance(exec_result, str) and exec_result.endswith(".md"):
                                preview_str = f"Query result saved to {os.path.basename(exec_result)}"
                            else:
                                preview_str = str(exec_result)
                            
                            if len(preview_str) > 200:
                                preview_str = preview_str[:197] + "..."
                            user_msg += f"\nExecution of new SQL successful. Result preview (1 row): {preview_str}\n"
                        user_msg += "Provide more /feedback or use /approved to save."
                        error_handler_module.display_response(user_msg)
                        return

                    except (ValidationError, json.JSONDecodeError, ValueError) as e:
                        if attempt == MAX_FEEDBACK_RETRIES:
                            error_handler_module.display_message(f"I'm having trouble processing your feedback. Error: {e}. Could you provide your feedback again, perhaps with more specific details?", level="ERROR")
                            return
                        feedback_prompt = f"Your previous response for updating the feedback report was invalid. Error: {e}. Please try again."
                    except Exception as e_gen:
                        if attempt == MAX_FEEDBACK_RETRIES:
                            error_handler_module.display_message(f"I encountered an issue while processing your feedback: {e_gen}. Could you rephrase your feedback?", level="ERROR")
                            return
                        feedback_prompt = "An unexpected error occurred. Please try to regenerate the updated feedback report JSON."
            else:
                error_handler_module.display_message("No SQL query generated yet for feedback. Use /generate_sql first.", level="ERROR")


        # Command: /approved
        elif base_command_lower == "approved":
            if not self.current_feedback_report_content:
                error_handler_module.display_message("No feedback report to approve. Use /generate_sql first.", level="ERROR")
                return

            print("Processing approval and updating memory, please wait...") # User-facing wait message
            try:
                # 1. Save Feedback Markdown
                feedback_filepath = memory_module.save_feedback_markdown(
                    self.current_feedback_report_content, 
                    self.current_db_name_identifier
                )
                # print(f"Feedback report saved to: {feedback_filepath}") # User sees final message

                saved_feedback_md_content = memory_module.read_feedback_file(feedback_filepath)
                if not saved_feedback_md_content:
                    error_handler_module.display_message("Could not read back saved feedback file for insights processing.", level="ERROR")
                    return

                insights_success = await insights_module.generate_and_update_insights(
                    self, 
                    saved_feedback_md_content,
                    self.current_db_name_identifier
                )
                if insights_success:
                    # print("Insights successfully generated/updated.") # User sees final message
                    self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
                else:
                    error_handler_module.display_message("Failed to generate or update insights from this feedback.", level="WARNING")
                
                if self.current_natural_language_question and self.current_feedback_report_content.final_corrected_sql_query:
                    try:
                        memory_module.save_nl2sql_pair(
                            self.current_db_name_identifier,
                            self.current_natural_language_question,
                            self.current_feedback_report_content.final_corrected_sql_query
                        )
                        # print("NLQ-SQL pair saved.") # User sees final message
                    except Exception as e_nl2sql:
                        error_handler_module.display_message(f"Failed to save NLQ-SQL pair: {e_nl2sql}", level="WARNING")
                else:
                    error_handler_module.display_message("Could not save NLQ-SQL pair due to missing NLQ or final SQL query in the report.", level="WARNING")

                final_user_message = f"Approved. Feedback report, insights, and NLQ-SQL pair for '{self.current_db_name_identifier}' saved."
                # Remove technical error message about insights update failure
                
                self._reset_feedback_cycle_state()
                self._reset_revision_cycle_state() # Also reset revision state on approval
                error_handler_module.display_response(f"{final_user_message}\nYou can start a new query with /generate_sql.")

            except Exception as e:
                error_handler_module.display_message(f"Error during approval process: {e}", level="ERROR")

        # Command: /revise
        elif base_command_lower == "revise":
            revision_prompt = argument_text
            if not revision_prompt:
                error_handler_module.display_message("Please provide a revision prompt after /revise.", level="ERROR")
                return
            
            sql_to_start_revision_with = None
            if self.is_in_revision_mode and self.current_revision_report_content and self.current_revision_report_content.final_revised_sql_query:
                sql_to_start_revision_with = self.current_revision_report_content.final_revised_sql_query
            elif self.current_feedback_report_content and self.current_feedback_report_content.final_corrected_sql_query:
                sql_to_start_revision_with = self.current_feedback_report_content.final_corrected_sql_query
            
            if not sql_to_start_revision_with:
                error_handler_module.display_message("No SQL query available to revise. Use /generate_sql first, or ensure a previous revision/feedback cycle completed with a query.", level="ERROR")
                return

            if not self.is_in_revision_mode:
                self.is_in_revision_mode = True
                # self.current_feedback_report_content = None # Decided against this to allow revising feedback results
                self.current_revision_report_content = RevisionReportContentModel(
                    initial_sql_for_revision=sql_to_start_revision_with
                )
            
            print(f"Revising SQL based on: \"{revision_prompt}\", please wait...")

            revision_history_for_llm_context = []
            if self.current_revision_report_content:
                for rev_iter in self.current_revision_report_content.revision_iterations:
                    revision_history_for_llm_context.append({"role": "user", "content": rev_iter.user_revision_prompt})
                    revision_history_for_llm_context.append({"role": "assistant", "content": f"Revised SQL: {rev_iter.revised_sql_attempt}\nExplanation: {rev_iter.revised_explanation}"})
            
            current_sql_for_this_iteration = self.current_revision_report_content.final_revised_sql_query or self.current_revision_report_content.initial_sql_for_revision
            
            revision_result = await revise_query_module.handle_revise_query_iteration(
                self,
                user_revision_prompt=revision_prompt,
                current_sql_to_revise=current_sql_for_this_iteration,
                revision_history_for_context=revision_history_for_llm_context,
                row_limit_for_preview=1 # Added row_limit_for_preview
            )

            if revision_result.get("revised_sql_query") and self.current_revision_report_content:
                new_iteration = RevisionIteration(
                    user_revision_prompt=revision_prompt,
                    revised_sql_attempt=revision_result["revised_sql_query"],
                    revised_explanation=revision_result.get("revised_explanation", "N/A")
                )
                self.current_revision_report_content.revision_iterations.append(new_iteration)
                self.current_revision_report_content.final_revised_sql_query = revision_result["revised_sql_query"]
                self.current_revision_report_content.final_revised_explanation = revision_result.get("revised_explanation", "N/A")
            
            message = revision_result.get("message_to_user", "Error in revision process.")
            if "Error" in message:
                error_handler_module.display_message(message, level="ERROR")
            else:
                error_handler_module.display_response(message)

        # Command: /approve_revision
        elif base_command_lower == "approve_revision":
            if not self.is_in_revision_mode or not self.current_revision_report_content or not self.current_revision_report_content.final_revised_sql_query:
                error_handler_module.display_message("No active revision cycle or final revised SQL to approve. Use /revise first.", level="ERROR")
                return

            # print("Finalizing revision and generating NLQ, please wait...") # User requested removal of such prints
            final_sql = self.current_revision_report_content.final_revised_sql_query
            
            nlq_gen_result = await revise_query_module.generate_nlq_for_revised_sql(
                self,
                final_revised_sql=final_sql,
                revision_report=self.current_revision_report_content
            )

            if nlq_gen_result.get("generated_nlq") and self.current_db_name_identifier and final_sql and self.current_revision_report_content:
                generated_nlq = nlq_gen_result["generated_nlq"]
                self.current_revision_report_content.llm_generated_nlq_for_final_sql = generated_nlq
                self.current_revision_report_content.llm_reasoning_for_nlq = nlq_gen_result.get("reasoning")
                
                user_msg_parts = [
                    f"Revision approved. Final SQL:\n```sql\n{final_sql}\n```",
                    f"LLM-generated Natural Language Question for this SQL:\n\"{generated_nlq}\"",
                    f"(Reasoning: {nlq_gen_result.get('reasoning', 'N/A')})"
                ]

                try:
                    memory_module.save_nl2sql_pair(
                        self.current_db_name_identifier,
                        generated_nlq,
                        final_sql
                    )
                    user_msg_parts.append(f"This NLQ-SQL pair has been saved for '{self.current_db_name_identifier}'.")

                    # --- Generate feedback report and insights ONLY IF feedback was used in revision ---
                    if self.feedback_used_in_current_revision_cycle and self.feedback_log_in_revision:
                        # user_msg_parts.append("Processing feedback from revision cycle for report and insights...") # Removed
                        
                        report_feedback_iterations = [
                            FeedbackIteration(
                                user_feedback_text=log_item["user_feedback_text"],
                                corrected_sql_attempt=log_item["corrected_sql_attempt"],
                                corrected_explanation=log_item["corrected_explanation"]
                            ) for log_item in self.feedback_log_in_revision
                        ]

                        # Construct a temporary FeedbackReportContentModel for analysis
                        temp_feedback_report_for_revision_analysis = FeedbackReportContentModel(
                            natural_language_question=generated_nlq,
                            initial_sql_query=self.feedback_log_in_revision[0]["sql_before_feedback"],
                            initial_explanation=self.feedback_log_in_revision[0]["explanation_before_feedback"],
                            feedback_iterations=report_feedback_iterations,
                            final_corrected_sql_query=self.feedback_log_in_revision[-1]["corrected_sql_attempt"], # Should be same as final_sql
                            final_explanation=self.feedback_log_in_revision[-1]["corrected_explanation"],
                        )
                        
                        feedback_model_schema_dict_for_analysis = FeedbackReportContentModel.model_json_schema()
                        feedback_model_schema_str_for_analysis = json.dumps(feedback_model_schema_dict_for_analysis, indent=2)
                        current_report_json_for_analysis = temp_feedback_report_for_revision_analysis.model_dump_json(indent=2)

                        analysis_prompt = (
                            f"You are an AI assistant. A user has gone through a query revision process, which included providing feedback that led to corrections. "
                            f"The final revised SQL has been approved, and its corresponding natural language question (NLQ) has been generated.\n"
                            f"NLQ for the final approved SQL: \"{generated_nlq}\"\n"
                            f"The state of the feedback report *before* your analysis (detailing feedback within the revision cycle) is:\n```json\n{current_report_json_for_analysis}\n```\n"
                            f"Your task is to populate the analytical fields of this feedback report: "
                            f"`why_initial_query_was_wrong_or_suboptimal` (comparing the 'initial_sql_query' in the report to the 'final_corrected_sql_query' based on the feedback), "
                            f"`why_final_query_works_or_is_improved` (explaining why the 'final_corrected_sql_query' is better due to the feedback), "
                            f"`database_insights_learned_from_this_query`, and `sql_lessons_learned_from_this_query`.\n\n"
                            f"Respond ONLY with the complete, updated JSON object for the `FeedbackReportContentModel`, conforming to this schema:\n"
                            f"```json\n{feedback_model_schema_str_for_analysis}\n```\n"
                            f"Ensure all fields from the input report JSON are preserved, and only the analysis fields are added/updated."
                        )
                        
                        MAX_ANALYSIS_RETRIES = 1
                        updated_feedback_report_from_revision: Optional[FeedbackReportContentModel] = None

                        for analysis_attempt in range(MAX_ANALYSIS_RETRIES + 1):
                            try:
                                analysis_messages = self.conversation_history + [{"role": "user", "content": analysis_prompt}]
                                response_format_analysis = None
                                if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                                    response_format_analysis = {"type": "json_object"}
                                
                                analysis_llm_response_obj = await self._send_message_to_llm(
                                    messages=analysis_messages, user_query=generated_nlq, response_format=response_format_analysis
                                )
                                analysis_llm_response_text, _ = await self._process_llm_response(analysis_llm_response_obj)

                                if analysis_llm_response_text.startswith("```json"): analysis_llm_response_text = analysis_llm_response_text[7:]
                                if analysis_llm_response_text.endswith("```"): analysis_llm_response_text = analysis_llm_response_text[:-3]
                                
                                updated_feedback_report_from_revision = FeedbackReportContentModel.model_validate_json(analysis_llm_response_text.strip())
                                break 
                            except (ValidationError, json.JSONDecodeError, ValueError) as e_analysis:
                                if analysis_attempt == MAX_ANALYSIS_RETRIES:
                                    error_handler_module.display_message(f"Failed to get LLM analysis for revision feedback report: {e_analysis}. Report will be saved without full analysis.", level="WARNING")
                                    updated_feedback_report_from_revision = temp_feedback_report_for_revision_analysis
                                    break
                                analysis_prompt = f"Your previous attempt to provide analysis for the feedback report was invalid. Error: {e_analysis}. Please try again."
                            except Exception as e_gen_analysis:
                                if analysis_attempt == MAX_ANALYSIS_RETRIES:
                                    error_handler_module.display_message(f"Unexpected error during LLM analysis for revision feedback report: {e_gen_analysis}. Report will be saved without full analysis.", level="WARNING")
                                    updated_feedback_report_from_revision = temp_feedback_report_for_revision_analysis
                                    break
                                analysis_prompt = "An unexpected error occurred during analysis. Please try again."
                        
                        if updated_feedback_report_from_revision:
                            try:
                                feedback_filepath_rev = memory_module.save_feedback_markdown(
                                    updated_feedback_report_from_revision, self.current_db_name_identifier
                                )
                                user_msg_parts.append(f"Feedback report for revision cycle saved: {os.path.basename(feedback_filepath_rev)}")

                                saved_feedback_md_content_rev = memory_module.read_feedback_file(feedback_filepath_rev)
                                if saved_feedback_md_content_rev:
                                    insights_success_rev = await insights_module.generate_and_update_insights(
                                        self, saved_feedback_md_content_rev, self.current_db_name_identifier
                                    )
                                    if insights_success_rev:
                                        user_msg_parts.append("Insights updated from revision feedback.")
                                        self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
                                    else:
                                        error_handler_module.display_message("Failed to update insights from revision feedback.", level="WARNING")
                                else: # saved_feedback_md_content_rev is None
                                    error_handler_module.display_message("Could not read back revision feedback file for insights processing.", level="WARNING")
                            except Exception as e_fb_insights_rev:
                                error_handler_module.display_message(f"Error during feedback report/insights saving for revision: {e_fb_insights_rev}", level="ERROR")
                        else: # updated_feedback_report_from_revision is None
                            error_handler_module.display_message("Could not generate the full feedback report for the revision cycle (LLM analysis failed).", level="WARNING")
                    elif not self.feedback_used_in_current_revision_cycle and self.current_revision_report_content:
                        # Generate insights directly from revision history
                        # user_msg_parts.append("Processing revision history for insights...") # Removed
                        try:
                            insights_from_revision_success = await revision_insights_module.generate_insights_from_revision_history(
                                self,
                                self.current_revision_report_content,
                                self.current_db_name_identifier
                            )
                            if insights_from_revision_success:
                                user_msg_parts.append("Insights successfully generated/updated from revision history.")
                                self.cumulative_insights_content = memory_module.read_insights_file(self.current_db_name_identifier)
                            else:
                                error_handler_module.display_message("Failed to generate or update insights from revision history.", level="WARNING")
                        except Exception as e_rev_ins:
                            error_handler_module.display_message(f"Error during revision insights generation: {e_rev_ins}", level="ERROR")
                    
                    user_msg_parts.append("You can start a new query with /generate_sql.")
                    self._reset_revision_cycle_state() 
                    error_handler_module.display_response("\n".join(user_msg_parts))
                
                except Exception as e_save:
                    error_handler_module.display_message(f"Error saving approved revision NLQ-SQL pair: {e_save}", level="ERROR")
            else: # Paired with: if nlq_gen_result.get("generated_nlq") ...
                message = nlq_gen_result.get("message_to_user", "Error: Could not generate NLQ for the revised SQL.")
                error_handler_module.display_message(message, level="ERROR")
        
        elif base_command_lower in ["list_commands", "help", "?"]:
            error_handler_module.display_response(self._get_commands_help_text())

        # If it's not a recognized command and starts with "/", it's unknown.
        elif query.startswith("/"):
            error_handler_module.display_message(f"Unknown command: '{query.split(None, 1)[0]}'. Current database: {self.current_db_name_identifier or 'None'}.\n"
                    f"Type /help, /? or /list_commands for available commands.", level="ERROR")
            return

        else:
            # If not a command (doesn't start with "/"), treat as navigation query
            # DO NOT reset feedback/revision state here.
            
            # Apply active table scope if it exists
            schema_for_query = self.db_schema_and_sample_data
            if self.active_table_scope is not None:
                schema_for_query = {k: v for k, v in self.db_schema_and_sample_data.items() if k in self.active_table_scope}

            nav_response = await database_navigation_module.handle_navigation_query(
                self, query, self.current_db_name_identifier, 
                self.cumulative_insights_content, schema_for_query
            )
            error_handler_module.display_response(nav_response)

    def _get_commands_help_text(self) -> str:
        """Returns the help text for commands."""
        return (
            "Available commands (use ':' or space after command, e.g., /generate_sql Your question):\n"
            "  /change_database {connection_string}\n"
            "    - Connects to a new PostgreSQL database. Example: /change_database postgresql://user:pass@host:port/dbname\n"
            "  /generate_sql {natural_language_question}\n"
            "    - Generates a SQL query based on your question.\n"
            "  /feedback {your_feedback_text}\n"
            "    - Provide feedback to refine the last SQL query.\n"
            "  /approved\n"
            "    - Approves the last SQL query from /generate_sql or /feedback. Saves report, insights, and NLQ-SQL pair.\n"
            "  /revise {your_revision_prompt}\n"
            "    - Iteratively revises the last SQL query.\n"
            "  /approve_revision\n"
            "    - Approves the final SQL from a /revise cycle. Generates an NLQ and saves the pair. Insights are generated based on revision history.\n"
            "  /reload_scope\n"
            "    - Reloads the table scope from the active_tables.txt file.\n"
            "  /change_model\n"
            "    - Interactively changes the LLM provider, API credentials, and model ID.\n"
            "  /change_profile\n"
            "    - Interactively changes the LLM profile.\n"
            "  /list_commands, /help, /?\n"
            "    - Shows this list of available commands.\n"
            "  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights."
        )

    async def chat_loop(self, initial_setup_done: bool):
        print(f"\n{Fore.MAGENTA}PostgreSQL Co-Pilot Started!{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Using LiteLLM with model: {Style.BRIGHT}{self.model_name}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Model provider: {Style.BRIGHT}{self.model_provider}{Style.RESET_ALL}")
        
        if initial_setup_done:
            print(f"\n{Fore.YELLOW}Welcome! Initial configuration complete.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Your settings have been saved to: {config_manager.get_config_file_path()}{Style.RESET_ALL}")
            print(self._get_commands_help_text())
        else:
            print(f"\n{Fore.CYAN}Type '/help', '/?' or '/list_commands' to see available commands.{Style.RESET_ALL}")
        
        if not self.is_initialized:
            print(f"\n{Fore.CYAN}Type a PostgreSQL connection string to begin, or use '/change_database'. Type 'quit' to exit.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.CYAN}Connected to '{self.current_db_name_identifier}'. Type a query or '/help' for commands. Type 'quit' to exit.{Style.RESET_ALL}")

        while True:
            user_input_query = ""
            try:
                prompt_text = f"\n{Fore.GREEN}[{self.current_db_name_identifier or f'{Style.DIM}No DB{Style.NORMAL}'}] {Style.BRIGHT}Query:{Style.RESET_ALL} "
                user_input_query = await asyncio.get_event_loop().run_in_executor(None, lambda: input(prompt_text).strip())
                
                if user_input_query.lower() == 'quit':
                    await self._cleanup_database_session(full_cleanup=True)
                    break
                if not user_input_query:
                    error_handler_module.display_response(self._get_commands_help_text())
                    continue
                
                await self.dispatch_command(user_input_query)

            except EOFError:
                error_handler_module.display_message("Exiting chat loop.", level="INFO", log=False)
                break
            except Exception as e:
                error_handler_module.handle_exception(e, user_query=user_input_query)

    async def cleanup(self):
        print("\nCleaning up client resources...")
        await self._cleanup_database_session(full_cleanup=True)
        await self.exit_stack.aclose()
        print("Client cleanup complete.")

async def main():
    colorama_init(autoreset=True)

    # --- Load Application Configuration & Initialize Logging ---
    initial_config_load = config_manager.load_config()
    app_config = config_manager.get_app_config()

    # Initialize logs immediately after getting config
    error_handler_module.initialize_logs(app_config.get("memory_base_dir"))
    
    initial_setup_was_performed = not initial_config_load

    if initial_setup_was_performed:
        while True:
            user_input = input("Type 'done' when you have finished editing the config file: ").strip().lower()
            if user_input == 'done':
                app_config = config_manager.get_app_config()
                break
            else:
                print("Please type 'done' to continue.")
    
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')

    # Determine the path to postgresql_server.py
    # It's assumed to be in the same directory as this chat script.
    try:
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        mcp_server_script = script_dir / "postgresql_server.py"

        if not mcp_server_script.exists():
            error_handler_module.display_message(f"MCP server script 'postgresql_server.py' not found in the expected location: {mcp_server_script}", level="FATAL")
        
        print(f"{Fore.BLUE}Located MCP server script at: {mcp_server_script}{Style.RESET_ALL}")

    except Exception as e:
        error_handler_module.display_message(f"Error determining MCP server script path: {e}", level="FATAL")

    custom_system_prompt = (
        "You are an expert PostgreSQL assistant. You will help users by generating SQL queries, "
        "analyzing feedback, and navigating database structures. Adhere strictly to JSON output formats when requested."
        "When you need to call a tool, you will be provided with a list of available tools. "
        "Your response should include a `tool_calls` field if you decide to use a tool. "
        "The `tool_calls` field should be a list of objects, each with `id`, `type: 'function'`, and `function` (containing `name` and `arguments` as a JSON string)."
    )
    # Specify the LiteLLM model name, e.g., "gpt-3.5-turbo", "gemini/gemini-pro", "claude-2"
    # This can also be set via an environment variable that LiteLLM reads, or passed as arg.
    # For now, using a Gemini model via LiteLLM as an example, assuming GOOGLE_API_KEY is set.
    # Or use "ollama/mistral" if ollama server is running with mistral.
    # Or "gpt-3.5-turbo" if OPENAI_API_KEY is set.
    # Defaulting to a common Gemini model that LiteLLM supports.
    # Model name is now read from .env inside LiteLLMMcpClient constructor
    client = LiteLLMMcpClient(app_config=app_config, system_instruction=custom_system_prompt) # Pass app_config

    # Ensure memory directories and logs are initialized before connecting to the MCP server
    memory_module.ensure_memory_directories()

    try:
        print(f"{Fore.BLUE}PostgreSQL Co-Pilot Client (LiteLLM) starting...{Style.RESET_ALL}")
        await client.connect_to_mcp_server(mcp_server_script)

        # --- Auto-initialize database connection from config ---
        if app_config.get("active_database_connection_string"):
            print(f"Found default database connection '{app_config.get('active_database_alias')}' in config. Attempting to connect...")
            conn_str = app_config["active_database_connection_string"]
            parsed_conn_str, parsed_db_name_id = client._extract_connection_string_and_db_name(conn_str)
            if parsed_conn_str and parsed_db_name_id:
                await client._handle_initialization(parsed_conn_str, parsed_db_name_id)
            else:
                error_handler_module.display_message(f"The default connection string '{app_config.get('active_database_alias')}' is invalid. Please check your config.json.", level="WARNING")
        # --- End Auto-initialize ---

        await client.chat_loop(initial_setup_done=initial_setup_was_performed)

        # Save conversation history (now client.conversation_history)
        if client.conversation_history:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use configured memory path
            conversation_history_base_dir = Path(app_config.get("memory_base_dir", memory_module.get_default_memory_base_path_text_for_chat_module())) / "conversation_history"
            conversation_history_base_dir.mkdir(parents=True, exist_ok=True)
            history_file_path = conversation_history_base_dir / f"conversation_litellm_{timestamp}.json"
            
            print(f"\n{Fore.CYAN}Saving conversation history to: {history_file_path}{Style.RESET_ALL}")
            try:
                with open(history_file_path, "w", encoding="utf-8") as f:
                    # Save as JSON for better structure, especially with tool calls
                    history_to_save = {
                        "db_name": client.current_db_name_identifier or "N/A",
                        "model_used": client.model_name,
                        "model_provider": client.model_provider,
                        "history": client.conversation_history
                    }
                    json.dump(history_to_save, f, indent=2)
                print(f"{Fore.GREEN}Conversation history saved.{Style.RESET_ALL}")
            except Exception as e_hist:
                error_handler_module.display_message(f"Error saving conversation history: {e_hist}", level="ERROR")
        else:
            error_handler_module.display_message("No conversation history to save.", level="WARNING", log=False)

    finally:
        await client.cleanup()

def entry_point_main():
    """Synchronous wrapper for the main async function, for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        error_handler_module.display_message("Application interrupted by user. Exiting...", level="INFO", log=False)
    except Exception as e:
        # Using the handler for top-level exceptions
        error_handler_module.handle_exception(e, context={"location": "entry_point_main"})
        sys.exit(1)


if __name__ == "__main__":
    entry_point_main()
