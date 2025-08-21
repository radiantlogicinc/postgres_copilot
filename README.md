# PostgreSQL Co-Pilot

PostgreSQL Co-Pilot is an intelligent assistant designed to help users interact with PostgreSQL databases. It leverages various Large Language Models (LLMs) via LiteLLM (including providers like OpenAI, Google Gemini, Anthropic, AWS Bedrock, DeepSeek, and OpenRouter) to understand natural language queries, generate SQL, process user feedback, and extract valuable insights about database schemas and query patterns. The Co-Pilot aims to streamline database interaction, making it more intuitive and efficient, especially for users who may not be SQL experts.

## Features

*   **Natural Language to SQL:** Converts user questions in plain English into executable PostgreSQL queries.
*   **Interactive Feedback Loop:** Allows users to provide feedback on generated SQL, which the Co-Pilot uses to refine queries and improve its understanding.
*   **Automated Insights Generation:** Learns from user interactions and feedback to build a cumulative knowledge base (`summarized_insights.md`) about database schemas, common query patterns, and SQL best practices.
*   **Database Schema Caching:** Stores database schema and sample data locally for faster subsequent interactions and offline reference.
*   **Conversation History:** Saves chat logs for review and context.
*   **Modular Design:** Separates concerns into distinct Python modules for initialization, SQL generation, insights management, database navigation, and memory operations.
*   **MCP Integration:** Utilizes the Model Context Protocol (MCP) to interact with a local PostgreSQL server, enabling database operations like query execution and schema fetching.

## Directory Structure

The application uses user-configurable paths for storing its data. During the first-run setup (or via `config.json`), you'll define:

1.  **Memory Base Directory** (e.g., `~/.local/share/PostgresCopilotTeam/PostgresCopilot/memory/`):
    This directory houses:
    ```
    memory_base_dir/
    ├── conversation_history/  # Stores chat logs
    ├── feedback/              # Stores detailed feedback reports (Markdown)
    ├── insights/              # Stores cumulative insights per database (Markdown)
    ├── schema/                # Stores cached database schemas and sample data (JSON)
    └── lancedb_stores/        # Base directory for LanceDB vector stores (one per database)
    ```

2.  **Approved Queries Directory** (e.g., `~/.local/share/PostgresCopilotTeam/PostgresCopilot/Approved_NL2SQL_Pairs/`):
    This directory stores the approved Natural Language Question - SQL query pairs as JSON files:
    ```
    approved_queries_dir/
    └── {database_identifier}_nl2sql_pairs.json
    ```

(Note: LLM provider choice, API keys/credentials, and model ID, along with these paths, are configured interactively during the first run of the application or via the `/change_model` command for LLM settings. All configurations are stored in a `config.json` file in a user-specific configuration directory like `~/.config/PostgresCopilot/` on Linux or `AppData/Roaming/PostgresCopilotTeam/PostgresCopilot/` on Windows.)

## Modules

*   **`postgres_copilot_chat.py`**: The main entry point and chat interface for interacting with the Co-Pilot. Manages the overall workflow, user commands, and communication with the Gemini model and MCP server.
*   **`memory_module.py`**: Handles all file and directory operations for storing and retrieving feedback, insights, schema data, and conversation history.
*   **`initialization_module.py`**: Manages the initial connection to a PostgreSQL database, fetches schema and sample data, and saves it.
*   **`sql_generation_module.py`**: Responsible for taking a natural language question, database schema, and existing insights to generate an initial SQL query using the Gemini model.
*   **`insights_module.py`**: Processes approved feedback reports to extract and update cumulative insights in `summarized_insights.md`.
*   **`database_navigation_module.py`**: Allows users to ask questions about the connected database structure using the cached schema and insights.
*   **`pydantic_models.py`**: Defines Pydantic models for structured data handling, such as feedback reports and insights.
*   **`postgresql_server.py`**: The MCP server script that directly interacts with the PostgreSQL database. It is automatically started and managed by the main `postgres-copilot` application.
*   **`model_change_module.py`**: Handles the `/change_model` command, allowing users to interactively update their LLM provider and settings.
*   **`config_manager.py`**: Manages user-specific configurations, including LLM settings and data storage paths, through a `config.json` file. Handles the interactive first-run setup.
*   **`pyproject.toml` & `uv.lock`**: Define project dependencies and manage the Python environment.

## Workflow

1.  **Initialization:**
    *   The user provides a PostgreSQL connection string (e.g., `postgresql://user:pass@host:port/dbname`) or uses the `/change_database` command.
    *   The `initialization_module` connects to the database, fetches its schema and some sample data.
    *   This information is saved locally in the `memory/schema/` directory by `memory_module.py`.
    *   Existing cumulative insights from `memory/insights/summarized_insights.md` are loaded.

2.  **SQL Generation:**
    *   The user asks a natural language question using the `/generate_sql` command (e.g., `/generate_sql: Show me all customers from California`).
    *   `sql_generation_module.py` uses the Gemini model, along with the cached schema and insights, to generate a SQL query.
    *   The generated SQL and an explanation are presented to the user.

3.  **Feedback (Optional):**
    *   If the SQL query is not perfect, the user can provide feedback using the `/feedback` command (e.g., `/feedback: The query should also filter by active status`).
    *   The Co-Pilot uses this feedback to generate a corrected SQL query and explanation. This can be an iterative process.

4.  **Approval & Insights Update:**
    *   Once the user is satisfied with the SQL query, they use the `/approved` command.
    *   `memory_module.py` saves a detailed feedback report (including the natural language question, initial SQL, iterations, final SQL, and LLM analysis) as a Markdown file in `memory/feedback/`.
    *   `insights_module.py` processes this feedback report. It uses the Gemini model to extract new insights or reinforce existing ones.
    *   These insights are then merged into the `memory/insights/summarized_insights.md` file.

5.  **Database Navigation:**
    *   Users can ask general questions about the database structure or data (without a `/` prefix).
    *   `database_navigation_module.py` attempts to answer these using the cached schema and cumulative insights.

## Setup and Installation

The PostgreSQL Co-Pilot is designed to be installed using `pipx` for a clean, isolated environment.

1.  **Prerequisites:**
    *   Python 3.10+
    *   `pipx` (Install it via `python -m pip install --user pipx` and then `python -m pipx ensurepath`)
    *   An accessible PostgreSQL database.

2.  **Build the Package (if installing from local source):**
    *   Navigate to the `postgres_copilot` directory (where `pyproject.toml` is located).
    *   Run: `python -m build`
    *   This will create a `.whl` file in the `postgres_copilot/dist/` directory.

3.  **Install using `pipx`:**
    *   **From local wheel file:**
        ```bash
        pipx install --force path/to/postgres_copilot/dist/db_copilot-0.1.0-py3-none-any.whl
        ```
        (Replace `path/to/` with the actual path, and ensure the version number matches the built wheel.)
    *   **From PyPI (once published):**
        ```bash
        pipx install db-copilot 
        ```

4.  **First-Run Configuration:**
    *   The first time you run `postgres-copilot` after installation, it will guide you through an interactive setup process.
    *   You will be asked to:
        *   Choose your preferred LLM provider (e.g., OpenAI, Google Gemini, Anthropic, AWS Bedrock, DeepSeek, OpenRouter).
        *   Enter the API key and any other required credentials for your chosen LLM.
        *   Specify the Model ID for the LLM.
        *   Confirm or change the default paths for storing:
            *   `memory` data (insights, schema, history, feedback, and vector stores).
            *   `Approved Queries` (NLQ-SQL JSON pair files).
    *   These settings are saved in a `config.json` file in your user-specific configuration directory.

## Usage

After installation, run the application by typing:

```bash
postgres-copilot
```

Once started, you can interact with the Co-Pilot using the following commands:

*   **Initial Connection:** Simply type or paste your PostgreSQL connection string:
    `postgresql://user:password@host:port/dbname`
*   **Change Database:**
    `/change_database: "your_new_connection_string"`
*   **Generate SQL:**
    `/generate_sql: Your natural language question about the database`
    (e.g., `/generate_sql: List all employees hired in the last year`)
*   **Provide Feedback:**
    `/feedback: Your comments or corrections for the last generated SQL`
    (e.g., `/feedback: The date format for hiring is YYYY-MM-DD, not MM/DD/YYYY`)
*   **Approve SQL:**
    `/approved`
    (Saves the feedback report, updates insights, and stores the NLQ-SQL pair)
*   **Revise SQL:**
    `/revise: Your instructions to modify the last SQL query`
*   **Approve Revision:**
    `/approve_revision`
    (Approves the final revised SQL, generates an NLQ for it, saves the pair, and updates insights)
*   **Change LLM Model:**
    `/change_model`
    (Allows you to interactively change the LLM provider, credentials, and model ID)
*   **Navigate/Ask Questions:**
    Type your question directly without a command prefix to ask about the database schema or general information based on loaded insights.
    (e.g., `What tables are related to products?`)
*   **List Commands / Help:**
    `/help` or `/list_commands` or `/?`
*   **Quit:**
    `quit`

## Dependencies

Key dependencies include:
*   `litellm`: For interacting with various Large Language Models.
*   `pipx`: Recommended for installation.
*   `appdirs`: For determining user-specific configuration/data directories.
*   `lancedb`: For vector storage and retrieval.
*   `sentence-transformers`: For generating embeddings.
*   `psycopg2`: PostgreSQL adapter for Python.
*   `pydantic`: For data validation and settings management.
*   `mcp` (Model Context Protocol): For communication with the internal PostgreSQL MCP server.
*   (See `pyproject.toml` for the full list)

## Future Enhancements

*   More sophisticated parsing of existing `summarized_insights.md` for merging new insights.
*   Support for other database systems beyond PostgreSQL.
*   Enhanced error handling and user guidance.
*   UI for easier interaction (e.g., web interface).
