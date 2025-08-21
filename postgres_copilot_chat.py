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
from prompt_toolkit import PromptSession, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import memory_module
import initialization_module
import sql_generation_module
import insights_module
import revision_insights_module # Added
import database_navigation_module
import revise_query_module # Added
import vector_store_module # Added for RAG
import hyde_feedback_module
import error_handler_module
from error_handler_module import handle_exception
import join_path_finder
import token_utils
from token_logging_module import log_token_usage
from token_tracking_module import TokenTracker
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
        
        # Initialize prompt_toolkit session
        history_file = os.path.join(app_config.get("memory_base_dir", ""), "input_history.txt")
        self.prompt_session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory()
        )
        
        # Create key bindings for multi-line input
        self.kb = KeyBindings()
        @self.kb.add(Keys.ControlJ)  # Ctrl+Enter to submit
        def _(event):
            event.current_buffer.validate_and_handle()
            
        # Add key bindings for proper cursor navigation
        @self.kb.add(Keys.Right)
        def _(event):
            # Move cursor right if not at the end of the buffer
            buffer = event.current_buffer
            if buffer.cursor_position < len(buffer.text):
                buffer.cursor_right()
                
        @self.kb.add(Keys.Backspace)
        def _(event):
            # Only allow backspace if there's text to delete
            buffer = event.current_buffer
            if buffer.cursor_position > 0:
                buffer.delete_before_cursor(1)
                
        @self.kb.add(Keys.Tab)
        def _(event):
            # Accept the current auto-suggestion when Tab is pressed
            buffer = event.current_buffer
            suggestion = buffer.suggestion
            if suggestion and suggestion.text:
                buffer.insert_text(suggestion.text)

        self.model_name = self.app_config.get("model_id") # Get from app_config
        self.provider = self.app_config.get("llm_provider")
        self.llm_api_key = self.app_config.get("api_key") # Store for potential direct use if needed
        
        # Initialize token tracker
        self.token_tracker = TokenTracker()
        
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
        # Main conversation history - will only contain system prompt
        self.conversation_history: list[Dict[str, Any]] = []
        # Module-specific conversation histories
        self.sql_generation_history: list[Dict[str, Any]] = []  # For SQL generation module
        self.feedback_revision_history: list[Dict[str, Any]] = []  # For feedback and revision
        self.navigation_history: list[Dict[str, Any]] = []  # For navigation responses only
        
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

    def _get_context_for_command(self, command_type: str) -> list:
        """Returns the appropriate conversation history based on command type."""
        # Start with system prompt
        base_context = [msg for msg in self.conversation_history if msg.get("role") == "system"]
        
        if command_type == "generate_sql":
            # For new SQL generation, just return system prompt
            return base_context
        elif command_type in ["feedback", "revise"]:
            # For feedback/revision, return system prompt + feedback history
            return base_context + self.feedback_revision_history
        elif command_type == "navigation":
            # For navigation, return system prompt + navigation history
            return base_context + self.navigation_history
        else:
            # Default to system prompt only
            return base_context
    
    def _reset_feedback_cycle_state(self):
        """Resets state for a new natural language query and its feedback cycle."""
        # print("Resetting feedback cycle state for new SQL generation...") # Internal detail
        self.current_natural_language_question = None
        self.current_feedback_report_content = None
        # Reset feedback history when starting a new feedback cycle
        self.feedback_revision_history = []
        # If starting a new feedback cycle, revision mode should also reset
        self._reset_revision_cycle_state()


    def _reset_revision_cycle_state(self):
        """Resets state for a new query revision cycle."""
        # print("Resetting revision cycle state...") # Internal detail
        self.current_revision_report_content = None
        self.is_in_revision_mode = False
        self.feedback_used_in_current_revision_cycle = False
        self.feedback_log_in_revision = []
        # Reset feedback/revision history when starting a new revision cycle
        self.feedback_revision_history = []
        
    def _reset_navigation_history(self):
        """Resets the navigation history."""
        self.navigation_history = []

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
                                  response_format: Optional[dict] = None, source: str = None, command_type: str = None) -> Any:
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
                
                # Get the actual tokens used in this specific call from the LLM response
                api_input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                llm_response = response.choices[0].message.content

                # Get current command from the stack trace
                current_command = None
                if 'sql_generation_module.py' in origin_script:
                    current_command = 'generate_sql'
                elif 'revise_query_module.py' in origin_script:
                    current_command = 'revise'
                elif 'hyde_module.py' in origin_script:
                    current_command = 'hyde'
                
                # Add tokens to tracker
                # We're tracking the actual API-reported tokens for accurate accounting
                # Remove is_incremental=True to ensure conversation totals are properly updated
                self.token_tracker.add_tokens(api_input_tokens, output_tokens, schema_tokens=schema_tokens, source=source)

                # Get insights tokens from the tracker for logging
                insights_tokens = self.token_tracker.current_command_insights_tokens
                
                # Get other prompt tokens from the tracker for logging
                other_prompt_tokens = self.token_tracker.current_command_other_prompt_tokens
                
                # Get hyde tokens and sql generation tokens for logging
                hyde_tokens = self.token_tracker.hyde_tokens
                sql_generation_tokens = self.token_tracker.sql_generation_tokens

                log_token_usage(
                    origin_script=origin_script,
                    origin_line=origin_line,
                    user_query=user_query,
                    prompt=prompt_text,
                    prompt_tokens=prompt_tokens,
                    schema_tokens=schema_tokens,
                    input_tokens=api_input_tokens,  # Using the API-reported input tokens
                    output_tokens=output_tokens,
                    llm_response=llm_response,
                    model_id=self.model_name,
                    command=current_command,
                    insights_tokens=insights_tokens,  # Use the tracked insights tokens
                    other_prompt_tokens=other_prompt_tokens,
                    hyde_tokens=hyde_tokens,
                    sql_generation_tokens=sql_generation_tokens,
                    source=source
                )
                
                # Add user message to appropriate history based on command type
                if messages[-1]["role"] == "user" and command_type:
                    if command_type == "generate_sql":
                        # For new SQL generation, don't add to main conversation history
                        # We'll only add the response if needed
                        pass
                    elif command_type in ["feedback", "revise"]:
                        # For feedback/revision, add to feedback history
                        self.feedback_revision_history.append(messages[-1])
                    elif command_type == "navigation":
                        # For navigation, don't add the schema-containing prompt to history
                        # We'll only add the response
                        pass
                    else:
                        # Default behavior - add to main history
                        self.conversation_history.append(messages[-1])
                elif messages[-1]["role"] == "user":
                    # If no command_type specified, use default behavior
                    self.conversation_history.append(messages[-1])

                return response
            except litellm.RateLimitError as e:
                # Log token usage on rate limit error
                prompt_text = messages[-1]['content']
                prompt_tokens = token_utils.count_tokens(prompt_text, self.model_name, self.provider)
                
                # For rate limit errors, estimate the total input tokens
                # This is an approximation since we don't have the API response
                estimated_input_tokens = 0
                for msg in messages:
                    estimated_input_tokens += token_utils.count_tokens(msg.get('content', ''), self.model_name, self.provider)
                
                # Ensure estimated_input_tokens is at least as large as schema_tokens
                if estimated_input_tokens < schema_tokens:
                    estimated_input_tokens = schema_tokens
                
                # Get insights tokens from the tracker for logging
                insights_tokens = self.token_tracker.current_command_insights_tokens

                log_token_usage(
                    origin_script=origin_script,
                    origin_line=origin_line,
                    user_query=user_query,
                    prompt=prompt_text,
                    prompt_tokens=prompt_tokens,
                    schema_tokens=schema_tokens,
                    input_tokens=estimated_input_tokens, # Estimate total input tokens
                    output_tokens=0, # No output tokens
                    llm_response=f"RateLimitError: {e}",
                    model_id=self.model_name,
                    insights_tokens=insights_tokens  # Use the tracked insights tokens
                )
                raise e # Re-raise the exception to be handled by the caller
        except Exception as e:
            # Do not log here for other exceptions; the caller is responsible for handling and logging.
            # Just add a placeholder to history and re-raise.
            if messages[-1]["role"] == "user" and messages[-1] not in self.conversation_history:
                 self.conversation_history.append(messages[-1])
            self.conversation_history.append({"role": "assistant", "content": "I'm having trouble processing your request."})
            raise # Re-raise the exception to be handled by the caller

    async def _process_llm_response(self, llm_response: Any, command_type: str = None) -> Tuple[str, bool]:
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

            # Add the assistant's message (that included the tool call request) to appropriate history
            if command_type:
                if command_type == "generate_sql":
                    # For SQL generation, don't add to main conversation history
                    pass
                elif command_type in ["feedback", "revise"]:
                    # For feedback/revision, add to feedback history
                    self.feedback_revision_history.append(assistant_message_for_history)
                    # Also add tool responses to feedback history
                    for resp in tool_call_responses_for_next_llm_call:
                        self.feedback_revision_history.append(resp)
                elif command_type == "navigation":
                    # For navigation, add to navigation history
                    self.navigation_history.append(assistant_message_for_history)
                    # Also add tool responses to navigation history
                    for resp in tool_call_responses_for_next_llm_call:
                        self.navigation_history.append(resp)
                else:
                    # Default behavior - add to main history
                    self.conversation_history.append(assistant_message_for_history)
                    # Also add tool responses to main history
                    for resp in tool_call_responses_for_next_llm_call:
                        self.conversation_history.append(resp)
            else:
                # If no command_type specified, use default behavior
                self.conversation_history.append(assistant_message_for_history)
                # Add all tool responses to history
                for resp in tool_call_responses_for_next_llm_call:
                    self.conversation_history.append(resp)

            # Make a follow-up call to LLM with tool responses
            # print("Sending tool responses back to LLM...") # Debug
            
            # The conversation history already contains:
            # ..., user_prompt, assistant_tool_call_request, tool_response_1, ...
            # So we can just send the current self.conversation_history
            
            # For follow-up call, use the same command_type
            follow_up_llm_response = await self._send_message_to_llm(
                self.conversation_history if not command_type else 
                (self._get_context_for_command(command_type) + 
                 (self.feedback_revision_history if command_type in ["feedback", "revise"] else 
                  self.navigation_history if command_type == "navigation" else [])),
                user_query, 
                tools=self.litellm_tools,
                command_type=command_type
            )
            
            # Process this new response (which should be the final text from LLM after tools)
            # This recursive call is safe as long as LLM doesn't loop infinitely on tool calls
            assistant_response_content, _ = await self._process_llm_response(follow_up_llm_response, command_type)
            # The _process_llm_response will handle adding the final assistant message to history.

        else: # No tool calls, just a direct text response
            if message.content: # Ensure there's content
                assistant_response_content = message.content
                # Add assistant's response to appropriate history based on command type
                assistant_message = {"role": "assistant", "content": assistant_response_content}
                if command_type:
                    if command_type == "generate_sql":
                        # For SQL generation, don't add to main conversation history
                        pass
                    elif command_type in ["feedback", "revise"]:
                        # For feedback/revision, add to feedback history
                        self.feedback_revision_history.append(assistant_message)
                    elif command_type == "navigation":
                        # For navigation, add response to navigation history
                        self.navigation_history.append(assistant_message)
                    else:
                        # Default behavior - add to main history
                        self.conversation_history.append(assistant_message)
                else:
                    # If no command_type specified, use default behavior
                    self.conversation_history.append(assistant_message)
            else: # Should not happen if the first check passed, but as a safeguard
                error_handler_module.display_message("LLM response message has no content.", level="ERROR")
                assistant_response_content = "Error: LLM response message has no content." # Keep for history
                self.conversation_history.append({"role": "assistant", "content": assistant_response_content})


        return assistant_response_content, tool_calls_made


    async def _handle_initialization(self, connection_string: str, db_name_id: str, force_regenerate: bool = False):
        """Handles the full DB initialization flow."""
        await self._cleanup_database_session(full_cleanup=True) # Clean slate for new DB
        
        success, message, schema_data = await initialization_module.perform_initialization(
            self, connection_string, db_name_id, force_regenerate=force_regenerate
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

    async def _handle_config_change(self, change_type: str):
        """Handles guiding the user to edit the config file for model or db changes."""
        config = config_manager.load_config()
        config_path = config_manager.get_config_file_path()
        display_path = config_manager.translate_path_for_display(str(config_path))

        if change_type == "model":
            profiles = config.get("llm_profiles", {})
            print("Available LLM profiles:")
            for alias, profile_data in profiles.items():
                print(f"- {alias} (Provider: {profile_data.get('provider')}, Model: {profile_data.get('model_id')})")
            print(f"\nTo change the active LLM, please edit the 'active_llm_profile_alias' in your config file.")
        
        elif change_type == "database":
            connections = config.get("database_connections", {})
            print("Available database connections:")
            for alias, conn_str in connections.items():
                print(f"- {alias}: {conn_str}")
            print(f"\nTo change the active database, please edit the 'active_database_alias' in your config file.")

        print(f"\nConfiguration file location: {display_path}")
        print("After saving your changes to the file, type 'done' and press Enter to reload.")

        while True:
            user_input = await self.prompt_session.prompt_async("", multiline=False)
            user_input = user_input.strip().lower()
            if user_input == 'done':
                break
            else:
                print("Please type 'done' to continue.")
        
        print("Reloading configuration...")
        self.app_config = config_manager.get_app_config()
        
        # Re-initialize based on the change type
        if change_type == "model":
            self.model_name = self.app_config.get("model_id")
            self.llm_api_key = self.app_config.get("api_key")
            # ... (rest of model change logic)
            self.conversation_history = []
            if self.system_instruction_content:
                self.conversation_history.append({"role": "system", "content": self.system_instruction_content})
            final_message = (
                f"Configuration reloaded. Active model is now: {self.model_name}\n"
                f"Conversation history has been reset."
            )
            error_handler_module.display_response(final_message)

        elif change_type == "database":
            conn_str = self.app_config.get("active_database_connection_string")
            db_alias = self.app_config.get("active_database_alias")
            if conn_str and db_alias:
                await self._handle_initialization(conn_str, db_alias, force_regenerate=False)
            else:
                error_handler_module.display_message("Could not find an active database in the reloaded configuration.", level="ERROR")

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
            await self._handle_config_change("model")
            return

        # Command: /change_database
        elif base_command_lower == "change_database":
            # This command now also uses the file-based approach
            await self._handle_config_change("database")
            return

        # Check for implicit initialization if not already initialized
        elif not self.is_initialized:
            parsed_conn_str, parsed_db_name_id = self._extract_connection_string_and_db_name(query) # Use original query for implicit check
            if parsed_conn_str:
                await self._handle_initialization(parsed_conn_str, parsed_db_name_id, force_regenerate=False)
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
            
        # Command: /regenerate_schema
        elif base_command_lower == "regenerate_schema":
            if not self.is_initialized or not self.current_db_name_identifier or not self.current_db_connection_string:
                error_handler_module.display_message("Please connect to a database first with /change_database.", level="ERROR")
                return
            
            print("Regenerating schema vectors and graph...")
            await self._handle_initialization(self.current_db_connection_string, self.current_db_name_identifier, force_regenerate=True)
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
            
            # Start token tracking for this command
            self.token_tracker.start_command("generate_sql")
            
            # Reset feedback and navigation history before generating new SQL
            self.feedback_revision_history = []
            self.navigation_history = []
            
            print("Generating SQL, please wait...")
            sql_gen_result_dict = await sql_generation_module.generate_sql_query(
                self, nl_question, self.db_schema_and_sample_data, self.cumulative_insights_content,
                row_limit_for_preview=1, # Ensure 1 row for preview from sql_generation_module
                command_type="generate_sql"
            )
            
            # End token tracking and get usage
            token_usage = self.token_tracker.end_command()
            
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
            
            # Add token usage information to the message
            if token_usage:
                token_usage_message = (
                    f"\n\nToken Usage for this command:\n"
                    f"  - Input tokens:    {token_usage['input_tokens']}\n"
                    f"    - Schema tokens:   {token_usage['schema_tokens']}\n"
                    f"    - Insights tokens: {token_usage['insights_tokens']}\n"
                    f"  - Output tokens:   {token_usage['output_tokens']}\n"
                    f"  - Total tokens for this command: {token_usage['total_tokens']}\n\n"
                    f"Conversation total: {token_usage['conversation_total']} tokens"
                )
                base_message_to_user += token_usage_message
            
            error_handler_module.display_response(base_message_to_user)

        # Command: /feedback
        elif base_command_lower == "feedback":
            user_feedback_text = argument_text
            if not user_feedback_text:
                error_handler_module.display_message("Please provide your feedback text after /feedback.", level="ERROR")
                return
                
            # Start token tracking for this command
            self.token_tracker.start_command("feedback")
            
            print("Processing feedback, please wait...")

            if self.is_in_revision_mode and self.current_revision_report_content and self.current_revision_report_content.final_revised_sql_query:
                # Apply feedback to the current revision
                current_sql = self.current_revision_report_content.final_revised_sql_query
                current_explanation = self.current_revision_report_content.final_revised_explanation or "N/A"
                
                # Retrieve schema context using HyDE
                hyde_context = ""
                table_names_from_hyde = []
                try:
                    hyde_context, table_names_from_hyde = await hyde_feedback_module.retrieve_hyde_feedback_context(
                        sql_query=current_sql,
                        user_feedback=user_feedback_text,
                        db_name_identifier=self.current_db_name_identifier,
                        llm_client=self
                    )
                except Exception as e_hyde:
                    handle_exception(e_hyde, user_feedback_text, {"context": "HyDE Context Retrieval for Feedback"})
                    hyde_context = "Failed to retrieve focused schema context via HyDE for feedback."
                
                # Load Schema Graph and Find Join Path
                schema_graph = memory_module.load_schema_graph(self.current_db_name_identifier)
                join_path_str = "No deterministic join path could be constructed for feedback."
                if schema_graph and table_names_from_hyde:
                    try:
                        join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
                        if join_clauses:
                            join_path_str = "\n".join(join_clauses)
                    except Exception as e_join:
                        handle_exception(e_join, user_feedback_text, {"context": "Join Path Finder for Feedback"})
                        join_path_str = "Error constructing join path for feedback."
                
                # Get insights context
                insights_context_str = "No cumulative insights provided."
                if self.cumulative_insights_content and self.cumulative_insights_content.strip():
                    insights_context_str = self.cumulative_insights_content
                
                # Prompt for correcting SQL based on feedback (within revision context)
                feedback_prompt_for_revision = (
                    f"You are an expert PostgreSQL SQL assistant. A user is providing feedback on a previously revised SQL query.\n"
                    f"CURRENT REVISED SQL QUERY:\n```sql\n{current_sql}\n```\n"
                    f"ITS EXPLANATION:\n{current_explanation}\n\n"
                    f"USER FEEDBACK: \"{user_feedback_text}\"\n\n"
                    f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the query):\n```\n{hyde_context}\n```\n\n"
                    f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{join_path_str}\n```\n\n"
                    f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
                    f"Based on this feedback and the schema information, please provide a corrected SQL query (must start with SELECT) and a brief explanation for the correction.\n"
                    f"Respond ONLY with a single JSON object matching this structure: "
                    f"{{ \"sql_query\": \"<Your corrected SELECT SQL query>\", \"explanation\": \"<Your explanation for the correction>\" }}\n"
                )
                MAX_FEEDBACK_RETRIES_IN_REVISION = 4
                corrected_sql_from_feedback = None
                corrected_explanation_from_feedback = None

                for attempt in range(MAX_FEEDBACK_RETRIES_IN_REVISION + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt_for_revision}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        # Count tokens for insights context
                        insights_tokens = token_utils.count_tokens(insights_context_str, self.model_name, self.provider)
                        
                        # Track insights tokens separately
                        self.token_tracker.add_insights_tokens(insights_tokens)
                        
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
                        # Check for errors in different formats
                        if isinstance(raw_output, str) and "Error:" in raw_output:
                            exec_error = raw_output
                        elif isinstance(raw_output, dict) and raw_output.get("status") == "error":
                            exec_error = raw_output.get("message", "Unknown execution error")
                            
                            # If there's an execution error, try to retrieve schema for the tables in the query
                            if exec_error:
                                try:
                                    # Extract table names from the query
                                    tables_in_query = sql_generation_module._extract_tables_from_query(corrected_sql_from_feedback)
                                    if tables_in_query:
                                        error_schema_context = []
                                        for table_name in tables_in_query:
                                            try:
                                                table_info_obj = await self.session.call_tool(
                                                    "describe_table", {"table_name": table_name}
                                                )
                                                table_info = self._extract_mcp_tool_call_output(table_info_obj, "describe_table")
                                                error_schema_context.append(json.dumps({table_name: table_info}, indent=2))
                                            except Exception as e_desc:
                                                handle_exception(e_desc, user_feedback_text, {"context": f"Describing table {table_name} for error recovery"})
                                        
                                        # If we got schema information, try to fix the query
                                        if error_schema_context:
                                            detailed_schema_context = "\n".join(error_schema_context)
                                            
                                            # Create a prompt to fix the query
                                            fix_prompt = (
                                                f"The SQL query generated based on user feedback resulted in an execution error.\n"
                                                f"SQL QUERY WITH ERROR:\n```sql\n{corrected_sql_from_feedback}\n```\n\n"
                                                f"EXECUTION ERROR MESSAGE:\n{exec_error}\n\n"
                                                f"USER FEEDBACK: \"{user_feedback_text}\"\n\n"
                                                f"CORRECT AND DETAILED SCHEMA:\n```json\n{detailed_schema_context}\n```\n\n"
                                                f"Please provide a corrected SQL query that fixes the error while still addressing the user's feedback. "
                                                f"For the explanation, describe how the *corrected* SQL query addresses the user's feedback. Do not mention the error or the fixing process.\n"
                                                f"Respond ONLY with a single JSON object: {{ \"sql_query\": \"<corrected SELECT query>\", \"explanation\": \"<explanation>\" }}"
                                            )
                                            
                                            # Send the prompt to the LLM
                                            fix_messages = self.conversation_history + [{"role": "user", "content": fix_prompt}]
                                            schema_tokens = token_utils.count_tokens(detailed_schema_context, self.model_name, self.provider)
                                            fix_llm_response_obj = await self._send_message_to_llm(fix_messages, user_feedback_text, schema_tokens)
                                            fix_text, _ = await self._process_llm_response(fix_llm_response_obj)
                                            
                                            if fix_text.startswith("```json"): fix_text = fix_text[7:]
                                            if fix_text.endswith("```"): fix_text = fix_text[:-3]
                                            fix_text = fix_text.strip()
                                            
                                            # Parse the response
                                            fixed_response = SQLGenerationResponse.model_validate_json(fix_text)
                                            if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                                # Update the SQL and explanation
                                                corrected_sql_from_feedback = fixed_response.sql_query
                                                corrected_explanation_from_feedback = fixed_response.explanation
                                                
                                                # Try executing the fixed query
                                                exec_obj = await self.session.call_tool("execute_postgres_query", {"query": corrected_sql_from_feedback, "row_limit": 1})
                                                raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                                                
                                                if isinstance(raw_output, str) and "Error:" in raw_output:
                                                    exec_error = raw_output
                                                elif isinstance(raw_output, dict) and raw_output.get("status") == "error":
                                                    exec_error = raw_output.get("message", "Unknown execution error")
                                                else:
                                                    exec_result = raw_output
                                                    exec_error = None
                                except Exception as e_recovery:
                                    handle_exception(e_recovery, user_feedback_text, {"context": "Error recovery for feedback"})
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
                    
                    # End token tracking and get usage
                    token_usage = self.token_tracker.end_command()
                    if token_usage:
                        user_msg += f"\nToken Usage for this feedback:\n" \
                                   f"Input tokens: {token_usage['input_tokens']}\n" \
                                   f"Output tokens: {token_usage['output_tokens']}\n" \
                                   f"Total tokens: {token_usage['total_tokens']}\n" \
                                   f"Conversation total: {token_usage['conversation_total']} tokens"
                    
                    user_msg += "\nUse `/revise Your new prompt`, more `/feedback`, or `/approve_revision`."
                    error_handler_module.display_response(user_msg)
                else:
                    error_handler_module.display_message("Failed to apply feedback to the revised query.", level="ERROR")

            elif self.current_feedback_report_content and self.current_natural_language_question:
                self._reset_revision_cycle_state() 
                current_report_json = self.current_feedback_report_content.model_dump_json(indent=2)
                feedback_model_schema_dict = FeedbackReportContentModel.model_json_schema()
                feedback_model_schema_str = json.dumps(feedback_model_schema_dict, indent=2)
                
                # Retrieve schema context using HyDE
                hyde_context = ""
                table_names_from_hyde = []
                try:
                    current_sql = self.current_feedback_report_content.final_corrected_sql_query
                    hyde_context, table_names_from_hyde = await hyde_feedback_module.retrieve_hyde_feedback_context(
                        sql_query=current_sql,
                        user_feedback=user_feedback_text,
                        db_name_identifier=self.current_db_name_identifier,
                        llm_client=self
                    )
                except Exception as e_hyde:
                    handle_exception(e_hyde, user_feedback_text, {"context": "HyDE Context Retrieval for Feedback"})
                    hyde_context = "Failed to retrieve focused schema context via HyDE for feedback."
                
                # Load Schema Graph and Find Join Path
                schema_graph = memory_module.load_schema_graph(self.current_db_name_identifier)
                join_path_str = "No deterministic join path could be constructed for feedback."
                if schema_graph and table_names_from_hyde:
                    try:
                        join_clauses = join_path_finder.find_join_path(table_names_from_hyde, schema_graph)
                        if join_clauses:
                            join_path_str = "\n".join(join_clauses)
                    except Exception as e_join:
                        handle_exception(e_join, user_feedback_text, {"context": "Join Path Finder for Feedback"})
                        join_path_str = "Error constructing join path for feedback."

                # Get insights context
                insights_context_str = "No cumulative insights provided."
                if self.cumulative_insights_content and self.cumulative_insights_content.strip():
                    insights_context_str = self.cumulative_insights_content
                
                feedback_prompt = (
                    f"You are refining a SQL query based on user feedback and updating a detailed report.\n"
                    f"The user's original question was: \"{self.current_natural_language_question}\"\n"
                    f"The current state of the feedback report (JSON format) is:\n```json\n{current_report_json}\n```\n"
                    f"The user has provided new feedback: \"{user_feedback_text}\"\n\n"
                    f"RELEVANT DATABASE SCHEMA INFORMATION (Use this to construct the query):\n```\n{hyde_context}\n```\n\n"
                    f"DETERMINISTIC JOIN PATH (You MUST use these exact JOIN clauses in your query if applicable):\n```\n{join_path_str}\n```\n\n"
                    f"CUMULATIVE INSIGHTS FROM PREVIOUS QUERIES (Use these to improve your query):\n```markdown\n{insights_context_str}\n```\n\n"
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
                
                MAX_FEEDBACK_RETRIES = 5  # Increased from 4 to 5 to match revision process
                for attempt in range(MAX_FEEDBACK_RETRIES + 1):
                    try:
                        messages_for_llm = self.conversation_history + [{"role": "user", "content": feedback_prompt}]
                        response_format = None
                        if self.model_name.startswith("gpt-") or self.model_name.startswith("openai/"):
                            response_format = {"type": "json_object"}
                        
                        # Count tokens for the schema context
                        schema_tokens = token_utils.count_tokens(hyde_context, self.model_name, self.provider)
                        
                        # Count tokens for insights context
                        insights_tokens = token_utils.count_tokens(insights_context_str, self.model_name, self.provider)
                        
                        # Track insights tokens separately
                        self.token_tracker.add_insights_tokens(insights_tokens)
                        
                        llm_response_obj = await self._send_message_to_llm(
                            messages=messages_for_llm,
                            user_query=user_feedback_text,
                            schema_tokens=schema_tokens,
                            response_format=response_format
                        )
                        response_text, _ = await self._process_llm_response(llm_response_obj)

                        if response_text.startswith("```json"): response_text = response_text[7:]
                        if response_text.endswith("```"): response_text = response_text[:-3]
                        
                        updated_report_model = FeedbackReportContentModel.model_validate_json(response_text.strip())
                        
                        if not updated_report_model.final_corrected_sql_query or \
                           not updated_report_model.final_corrected_sql_query.strip().upper().startswith("SELECT"):
                            raise ValueError("Corrected SQL in feedback report must start with SELECT.")

                        # Store the original model and SQL before execution validation
                        original_report_model = updated_report_model
                        original_sql = updated_report_model.final_corrected_sql_query
                        original_explanation = updated_report_model.final_explanation
                        
                        # Execute the SQL to validate it
                        exec_result, exec_error = None, None
                        try:
                            exec_obj = await self.session.call_tool("execute_postgres_query", {"query": updated_report_model.final_corrected_sql_query, "row_limit": 1})
                            raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                            # Check for errors in different formats
                            if isinstance(raw_output, str) and "Error:" in raw_output:
                                exec_error = raw_output
                            elif isinstance(raw_output, dict) and raw_output.get("status") == "error":
                                exec_error = raw_output.get("message", "Unknown execution error")
                                
                                # If there's an execution error, try to retrieve schema for the tables in the query
                                if exec_error and attempt < MAX_FEEDBACK_RETRIES:
                                    try:
                                        # Extract table names from the query
                                        tables_in_query = sql_generation_module._extract_tables_from_query(updated_report_model.final_corrected_sql_query)
                                        if tables_in_query:
                                            error_schema_context = []
                                            for table_name in tables_in_query:
                                                try:
                                                    table_info_obj = await self.session.call_tool(
                                                        "describe_table", {"table_name": table_name}
                                                    )
                                                    table_info = self._extract_mcp_tool_call_output(table_info_obj, "describe_table")
                                                    error_schema_context.append(json.dumps({table_name: table_info}, indent=2))
                                                except Exception as e_desc:
                                                    handle_exception(e_desc, user_feedback_text, {"context": f"Describing table {table_name} for error recovery"})
                                            
                                            # If we got schema information, try to fix the query
                                            if error_schema_context:
                                                detailed_schema_context = "\n".join(error_schema_context)
                                                
                                                # Create a prompt to fix the query
                                                fix_prompt = (
                                                    f"The SQL query generated based on user feedback resulted in an execution error.\n"
                                                    f"SQL QUERY WITH ERROR:\n```sql\n{updated_report_model.final_corrected_sql_query}\n```\n\n"
                                                    f"EXECUTION ERROR MESSAGE:\n{exec_error}\n\n"
                                                    f"USER FEEDBACK: \"{user_feedback_text}\"\n\n"
                                                    f"CORRECT AND DETAILED SCHEMA:\n```json\n{detailed_schema_context}\n```\n\n"
                                                    f"Please provide a corrected SQL query that fixes the error while still addressing the user's feedback. "
                                                    f"For the explanation, describe how the *corrected* SQL query addresses the user's feedback. Do not mention the error or the fixing process.\n"
                                                    f"Respond ONLY with a single JSON object: {{ \"sql_query\": \"<corrected SELECT query>\", \"explanation\": \"<explanation>\" }}"
                                                )
                                                
                                                # Send the prompt to the LLM
                                                fix_messages = self.conversation_history + [{"role": "user", "content": fix_prompt}]
                                                schema_tokens = token_utils.count_tokens(detailed_schema_context, self.model_name, self.provider)
                                                fix_llm_response_obj = await self._send_message_to_llm(fix_messages, user_feedback_text, schema_tokens)
                                                fix_text, _ = await self._process_llm_response(fix_llm_response_obj)
                                                
                                                if fix_text.startswith("```json"): fix_text = fix_text[7:]
                                                if fix_text.endswith("```"): fix_text = fix_text[:-3]
                                                fix_text = fix_text.strip()
                                                
                                                # Parse the response
                                                fixed_response = SQLGenerationResponse.model_validate_json(fix_text)
                                                if fixed_response.sql_query and fixed_response.sql_query.strip().upper().startswith("SELECT"):
                                                    # Update the SQL and explanation in the report model
                                                    updated_report_model.final_corrected_sql_query = fixed_response.sql_query
                                                    updated_report_model.final_explanation = fixed_response.explanation
                                                    
                                                    # Update the latest feedback iteration as well
                                                    if updated_report_model.feedback_iterations and len(updated_report_model.feedback_iterations) > 0:
                                                        latest_iteration = updated_report_model.feedback_iterations[-1]
                                                        latest_iteration.corrected_sql_attempt = fixed_response.sql_query
                                                        latest_iteration.corrected_explanation = fixed_response.explanation
                                                    
                                                    # Try executing the fixed query
                                                    exec_obj = await self.session.call_tool("execute_postgres_query", {"query": fixed_response.sql_query, "row_limit": 1})
                                                    raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                                                    
                                                    if isinstance(raw_output, str) and "Error:" in raw_output:
                                                        exec_error = raw_output
                                                        # Revert to original if still error
                                                        updated_report_model = original_report_model
                                                    elif isinstance(raw_output, dict) and raw_output.get("status") == "error":
                                                        exec_error = raw_output.get("message", "Unknown execution error")
                                                        # Revert to original if still error
                                                        updated_report_model = original_report_model
                                                    else:
                                                        exec_result = raw_output
                                                        exec_error = None
                                                        # Keep the fixed version
                                    except Exception as e_recovery:
                                        handle_exception(e_recovery, user_feedback_text, {"context": "Error recovery for feedback"})
                                        # Revert to original if recovery failed
                                        updated_report_model = original_report_model
                            else:
                                exec_result = raw_output
                        except Exception as e_exec:
                            exec_error = str(e_exec)
                            
                        # If we still have an error after all attempts to fix, revert to original
                        if exec_error and attempt == MAX_FEEDBACK_RETRIES:
                            updated_report_model = original_report_model
                            exec_obj = await self.session.call_tool("execute_postgres_query", {"query": original_sql, "row_limit": 1})
                            raw_output = self._extract_mcp_tool_call_output(exec_obj, "execute_postgres_query")
                            if not (isinstance(raw_output, str) and "Error:" in raw_output) and not (isinstance(raw_output, dict) and raw_output.get("status") == "error"):
                                exec_result = raw_output
                                exec_error = None
                        
                        self.current_feedback_report_content = updated_report_model

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
                        
                        # End token tracking and get usage
                        token_usage = self.token_tracker.end_command()
                        if token_usage:
                            token_usage_message = (
                                f"\n\nToken Usage for this command:\n"
                                f"  - Input tokens:    {token_usage['input_tokens']}\n"
                                f"    - Schema tokens:   {token_usage['schema_tokens']}\n"
                                f"    - Insights tokens: {token_usage['insights_tokens']}\n"
                                f"  - Output tokens:   {token_usage['output_tokens']}\n"
                                f"  - Total tokens for this command: {token_usage['total_tokens']}\n\n"
                                f"Conversation total: {token_usage['conversation_total']} tokens"
                            )
                            user_msg += token_usage_message
                        
                        user_msg += "\nProvide more /feedback or use /approved to save."
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

            # Start token tracking for this command
            self.token_tracker.start_command("approved")
            
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
                
                # End token tracking and get usage
                token_usage = self.token_tracker.end_command()
                if token_usage:
                    token_usage_message = (
                        f"\n\nToken Usage for this command:\n"
                        f"  - Input tokens:    {token_usage['input_tokens']}\n"
                        f"    - Schema tokens:   {token_usage['schema_tokens']}\n"
                        f"    - Insights tokens: {token_usage['insights_tokens']}\n"
                        f"  - Output tokens:   {token_usage['output_tokens']}\n"
                        f"  - Total tokens for this command: {token_usage['total_tokens']}\n\n"
                        f"Conversation total: {token_usage['conversation_total']} tokens"
                    )
                    final_user_message += token_usage_message
                
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
                
            # Start token tracking for this command
            self.token_tracker.start_command("revise")
            
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
                insights_markdown_content=self.cumulative_insights_content, # Pass insights
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
            
            # End token tracking and get usage
            token_usage = self.token_tracker.end_command()
            if token_usage and "Error" not in message:
                token_usage_message = (
                    f"\n\nToken Usage for this command:\n"
                    f"  - Input tokens:    {token_usage['input_tokens']}\n"
                    f"    - Schema tokens:   {token_usage['schema_tokens']}\n"
                    f"    - Insights tokens: {token_usage['insights_tokens']}\n"
                    f"  - Output tokens:   {token_usage['output_tokens']}\n"
                    f"  - Total tokens for this command: {token_usage['total_tokens']}\n\n"
                    f"Conversation total: {token_usage['conversation_total']} tokens"
                )
                message += token_usage_message
                
            if "Error" in message:
                error_handler_module.display_message(message, level="ERROR")
            else:
                error_handler_module.display_response(message)

        # Command: /approve_revision
        elif base_command_lower == "approve_revision":
            if not self.is_in_revision_mode or not self.current_revision_report_content or not self.current_revision_report_content.final_revised_sql_query:
                error_handler_module.display_message("No active revision cycle or final revised SQL to approve. Use /revise first.", level="ERROR")
                return
                
            # Start token tracking for this command
            self.token_tracker.start_command("approve_revision")

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
                    
                    # End token tracking and get usage
                    token_usage = self.token_tracker.end_command()
                    if token_usage:
                        token_usage_message = (
                            f"\n\nToken Usage for this command:\n"
                            f"  - Input tokens:    {token_usage['input_tokens']}\n"
                            f"    - Schema tokens:   {token_usage['schema_tokens']}\n"
                            f"    - Insights tokens: {token_usage['insights_tokens']}\n"
                            f"  - Output tokens:   {token_usage['output_tokens']}\n"
                            f"  - Total tokens for this command: {token_usage['total_tokens']}\n\n"
                            f"Conversation total: {token_usage['conversation_total']} tokens"
                        )
                        user_msg_parts.append(token_usage_message)
                    
                    user_msg_parts.append("You can start a new query with /generate_sql.")
                    self._reset_revision_cycle_state() 
                    error_handler_module.display_response("\n".join(user_msg_parts))
                
                except Exception as e_save:
                    error_handler_module.display_message(f"Error saving approved revision NLQ-SQL pair: {e_save}", level="ERROR")
            else: # Paired with: if nlq_gen_result.get("generated_nlq") ...
                message = nlq_gen_result.get("message_to_user", "Error: Could not generate NLQ for the revised SQL.")
                error_handler_module.display_message(message, level="ERROR")
        
        # Command: /navigate_db
        elif base_command_lower == "navigate_db":
            user_query = argument_text
            if not user_query:
                error_handler_module.display_message("Please provide a query after /navigate_db.", level="ERROR")
                return
            
            # Call the new handle_navigate_db_command function
            nav_response = await database_navigation_module.handle_navigate_db_command(
                self, user_query, self.current_db_name_identifier, 
                self.cumulative_insights_content
            )
            error_handler_module.display_response(nav_response)
            
        elif base_command_lower in ["list_commands", "help", "?"]:
            help_text = self._get_commands_help_text()
            
            # Add token usage information if available
            if self.token_tracker.conversation_total_input_tokens > 0 or self.token_tracker.conversation_total_output_tokens > 0:
                token_usage_message = f"\n\nCurrent Token Usage:\n{self.token_tracker.get_command_history_message()}"
                help_text += token_usage_message
                
            error_handler_module.display_response(help_text)

        # If it's not a recognized command and starts with "/", it's unknown.
        elif query.startswith("/"):
            error_handler_module.display_message(f"Unknown command: '{query.split(None, 1)[0]}'. Current database: {self.current_db_name_identifier or 'None'}.\n"
                    f"Type /help, /? or /list_commands for available commands.", level="ERROR")
            return

        else:
            # If not a command (doesn't start with "/"), treat as navigation query
            # DO NOT reset feedback/revision state here.
            
            # Do not pass schema to database navigation module
            # It will fetch table names directly from the database
            nav_response = await database_navigation_module.handle_navigation_query(
                self, query, self.current_db_name_identifier, 
                self.cumulative_insights_content
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
            "  /regenerate_schema\n"
            "    - Forces regeneration of schema vectors and graph even if they already exist.\n"
            "  /navigate_db {your_query}\n"
            "    - Generates SQL to answer your question and provides a descriptive answer based on the data.\n"
            "  /change_model\n"
            "    - Guides you to change the active LLM profile in the config file.\n"
            "  /change_database\n"
            "    - Guides you to change the active database connection in the config file.\n"
            "  /list_commands, /help, /?\n"
            "    - Shows this list of available commands.\n"
            "  Or, type a natural language question without a '/' prefix to ask about the current database schema or insights.\n\n"
            "Key Bindings:\n"
            "  - Enter: Create a new line\n"
            "  - Ctrl+Enter: Submit your query or command"
        )

    async def chat_loop(self, initial_setup_done: bool):
        print(f"\n{Fore.MAGENTA}PostgreSQL Co-Pilot Started!{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Using LiteLLM with model: {Style.BRIGHT}{self.model_name}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Model provider: {Style.BRIGHT}{self.model_provider}{Style.RESET_ALL}")
        
        if initial_setup_done:
            print(f"\n{Fore.YELLOW}Welcome! Initial configuration complete.{Style.RESET_ALL}")
            config_path = config_manager.get_config_file_path()
            display_path = config_manager.translate_path_for_display(str(config_path))
            print(f"{Fore.YELLOW}Your settings have been saved to: {display_path}{Style.RESET_ALL}")
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
                # Create a prompt with green color for the database name and query text using prompt_toolkit's HTML formatting
                prompt_text = HTML(f"\n<ansigreen>[{self.current_db_name_identifier or 'No DB'}] Query: </ansigreen>")
                # Use prompt_toolkit for multi-line input
                user_input_query = await self.prompt_session.prompt_async(
                    prompt_text,
                    key_bindings=self.kb,
                    multiline=True
                )
                user_input_query = user_input_query.strip()
                
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
    # Initialize colorama with autoreset=True to ensure proper color handling
    colorama_init(autoreset=True)

    # --- Load Application Configuration & Initialize Logging ---
    initial_config_load = config_manager.load_config()
    app_config = config_manager.get_app_config()

    # Initialize logs immediately after getting config
    error_handler_module.initialize_logs(app_config.get("memory_base_dir"))
    
    initial_setup_was_performed = not initial_config_load

    if initial_setup_was_performed:
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(None, lambda: input("Type 'done' when you have finished editing the config file: ").strip().lower())
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
                await client._handle_initialization(parsed_conn_str, parsed_db_name_id, force_regenerate=False)
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
            
            display_path = config_manager.translate_path_for_display(str(history_file_path))
            print(f"\n{Fore.CYAN}Saving conversation history to: {display_path}{Style.RESET_ALL}")
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
