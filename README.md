# PostgreSQL Co-Pilot

PostgreSQL Co-Pilot is an intelligent, command-line assistant designed to run in Docker. It helps you interact with your PostgreSQL databases using natural language. It leverages various Large Language Models (LLMs) to translate questions into SQL, process feedback to refine queries, and build a cumulative knowledge base about your database.

The entire application is self-contained in a Docker image, requiring only Docker and Docker Compose to be installed on your system.

## Quick Start

Get up and running in a few simple steps. All you need is **Docker** and **Docker Compose** installed.

1.  **Download the necessary files:**

    In your terminal, run the following commands to download the `run.sh` script and the `client-docker-compose.yml` file.

    ```bash
    # Download the run script
    curl -L "https://gist.githubusercontent.com/svyas113/75bdc62d82b00be29a7f6f3a291ab0b4/raw/run.sh" -o run.sh

    # Download the Docker Compose file
    curl -L "https://gist.githubusercontent.com/svyas113/75bdc62d82b00be29a7f6f3a291ab0b4/raw/client-docker-compose.yml" -o client-docker-compose.yml
    ```

2.  **Make the script executable:**

    ```bash
    chmod +x ./run.sh
    ```

3.  **Run the application:**

    ```bash
    ./run.sh
    ```

    That's it\! The script will automatically create a local `./data` directory for your configuration and memory, pull the latest Docker image, and launch the application.

-----

## First-Time Configuration

The very first time you run `./run.sh`, the Co-Pilot will guide you through an initial setup process:

1.  A new configuration file will be created at **`./data/config/config.json`**.
2.  The application will print the path to this file and ask you to edit it.
3.  Open `./data/config/config.json` in your favorite text editor.
4.  Fill in your LLM provider details (API key, model, etc.) and your database connection string.
5.  Save the file and return to your terminal. Type `done` and press Enter to continue.

### Example `config.json`

Below is an example of what your completed `config.json` file should look like. You will need to replace `"YOUR_GEMINI_API_KEY"` and the database connection details with your own.

```json
{
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
    "memory_base_dir": "/app/data/memory",
    "approved_queries_dir": "/app/data/approved_NL2SQL_Pairs",
    "nl2sql_vector_store_base_dir": "/app/data/memory/lancedb_stores"
}
```

**Important:** To connect to a PostgreSQL database running on your **host machine** (your laptop/desktop), you must use `host.docker.internal` as the hostname in the connection string, as shown in the example.

## How It Works

The `run.sh` script simplifies the entire setup process:

  * It creates a local `data` directory to permanently store all your configurations, database schemas, feedback history, and insights.
  * It ensures your local user owns these files, avoiding permission issues.
  * It pulls the latest `postgres-copilot2` Docker image.
  * It starts the application using `docker-compose -f client-docker-compose.yml`, linking your local `data` directory to the container's `/app/data` directory.

This means all application data persists on your machine in the `./data` folder, even though the application itself runs inside a temporary Docker container.

## Usage

Once started and configured, interact with the Co-Pilot using these commands:

  * **Generate SQL:** `/generate_sql Your natural language question`
  * **Provide Feedback:** `/feedback Your comments or corrections`
  * **Approve SQL:** `/approved`
  * **Revise a Query:** `/revise Your instructions to modify the last query`
  * **Approve a Revision:** `/approve_revision`
  * **Ask about Schema:** `What tables are related to products?` (no command needed)
  * **Help:** `/help`, `/list_commands`, or `/?`
  * **Quit:** `quit`

## Features

  * **Natural Language to SQL:** Converts plain English into executable PostgreSQL queries.
  * **Interactive Feedback & Revision:** Refine queries iteratively with simple feedback.
  * **Automated Insights & RAG:** Learns from your interactions and uses a vector store to find similar past queries, dramatically improving accuracy.
  * **Multi-LLM Support:** Configure different LLM providers like Google Gemini, OpenAI, Anthropic, and more in your `config.json`.
  * **Persistent Memory:** All configurations, schemas, and learnings are stored locally in your `./data` directory.
