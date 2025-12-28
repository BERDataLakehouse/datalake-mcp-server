# Using BERDL Datalake MCP Server with Claude Code

Claude Code is Anthropic's official CLI tool that brings Claude's capabilities directly to your command line. This guide shows you how to connect Claude Code to the BERDL Datalake MCP Server to query your data lake using natural language.

## Prerequisites

Before you begin, you'll need:

1. **BERDL_USER Role** - Your KBase account must be added to the BERDL_USER group
   - Contact the KBase system admin team to request access
   - This role is required to authenticate with the BERDL MCP server

2. **Claude Code installed** - Follow instructions at https://claude.ai/code

3. **KBase Auth Token** - Your KBASE_AUTH_TOKEN for authentication

4. **Access to BERDL JupyterHub** - Required for user provisioning
   - You must be able to log in to BERDL JupyterHub
   - See the [BERDL JupyterHub User Guide](https://github.com/BERDataLakehouse/spark_notebook/blob/main/docs/user_guide.md) for login instructions

> **💡 Important: Log in to BERDL JupyterHub First**
>
> Before using the MCP server with Claude Code, you must log in to BERDL JupyterHub at least once. This ensures:
> - Your user account is properly provisioned
> - Your MinIO buckets are created
> - Permissions and policies are configured
> - Namespace isolation is set up
>
> Without this initial login, the MCP server may not be able to access your data.

> **⚡ Performance Tip: Stay Logged in to JupyterHub**
>
> While not required for the MCP server to work, we **strongly recommend keeping an active JupyterHub session** while using Claude Code:
> - **Faster queries:** When logged in, we create a dedicated user Spark cluster just for you
> - **Better performance:** Your dedicated cluster provides faster query execution
> - **Automatic fallback:** If not logged in, the MCP server automatically switches to a shared static Spark cluster, which may be slower
>
> To maintain your session: Keep a JupyterHub browser tab open while using Claude Code.

## Quick Start

### Step 1: Get Your KBase Auth Token

**Option A: From BERDL JupyterHub (Recommended)**

If you're logged into BERDL JupyterHub, your token is available as an environment variable:

1. Open any notebook in BERDL JupyterHub
2. Run the following code:

```python
import os
token = os.environ["KBASE_AUTH_TOKEN"]
print(f"Your token: {token}")
```

**Option B: From KBase Narrative**

1. Log in to https://narrative.kbase.us/
2. Open any narrative or create a new one
3. Add a code cell and run:

```python
import os
token = os.environ["KB_AUTH_TOKEN"]
print(f"Your token: {token}")
```

### Step 2: Connect Claude Code to BERDL MCP Server

Use the `claude mcp add` command to register the BERDL Datalake MCP server:

```bash
claude mcp add --transport http --scope user berdl-datalake \
  https://hub.berdl.kbase.us/apis/mcp/mcp \
  --header "Authorization: Bearer YOUR_KBASE_AUTH_TOKEN"
```

Replace `YOUR_KBASE_AUTH_TOKEN` with your actual token from Step 1.

**Command breakdown:**
- `--transport http` - Use Streamable HTTP transport protocol (enables horizontal scaling)
- `--scope user` - Register for current user only
- `berdl-datalake` - Name for this MCP server connection
- `https://hub.berdl.kbase.us/apis/mcp/mcp` - BERDL MCP server endpoint
- `--header "Authorization: Bearer ..."` - Your authentication header

### Step 3: Verify Connection

Check that the MCP server is connected and healthy:

```bash
claude mcp list
```

You should see output similar to:

```
Checking MCP server health...

berdl-datalake: https://hub.berdl.kbase.us/apis/mcp/mcp (HTTP) - ✓ Connected
```

### Step 4: Verify Configuration File

Your MCP server configuration is stored in `~/.claude.json`. You can verify it contains:

```json
{
  "mcpServers": {
    "berdl-datalake": {
      "type": "http",
      "url": "https://hub.berdl.kbase.us/apis/mcp/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_KBASE_AUTH_TOKEN"
      }
    }
  }
}
```

> **🔒 Security Note**
>
> Your `~/.claude.json` file contains your authentication token. Ensure it has restricted permissions:
> ```bash
> chmod 600 ~/.claude.json
> ```

### Step 5: Start Using Claude Code

Now you can use Claude Code to query your data lake with natural language! Start Claude Code:

```bash
claude
```

## Example Queries

Once Claude Code is running, you can use natural language to interact with your data:

### Discover Databases
```
What databases do I have access to?
```

### List Tables
```
What tables are in kbase database?
```

### Explore Table Schema
```
Show me the schema of the kbase_ke_pangenome table in kbase database
```

### Query Data
```
Show me 10 sample rows from kbase.kbase_ke_pangenome
```

## Available MCP Tools

Claude Code has access to the following tools through the BERDL MCP server:

| Tool | Description | Example Query |
|------|-------------|---------------|
| **list_databases** | List all databases you can access | "What databases exist?" |
| **list_database_tables** | List tables in a database | "What tables are in my_db?" |
| **get_table_schema** | Get column names and types | "Show schema of my_table" |
| **get_database_structure** | Get full database/table hierarchy | "Give me complete database structure" |
| **count_delta_table** | Count rows in a table | "How many rows in my_table?" |
| **sample_delta_table** | Preview table data | "Show 10 rows from my_table" |
| **query_delta_table** | Execute SQL queries | "Find top 10 users by activity" |
| **select_delta_table** | Structured SELECT queries | Used internally by Claude |


## Best Practices

### 1. Always Start with Discovery
```
# First, see what you have access to
What databases can I access?

# Then explore tables
What tables are in kbase database?

# Check schema before querying
Show me the schema of the kbase_ke_pangenome table
```

### 2. Be Specific with Database/Table Names
```
# ✅ Good - explicit database.table
Show me data from kbase.kbase_ke_pangenome

# ❌ Less clear - ambiguous
Show me pangenome data
```

### 3. Use Limits for Large Datasets
```
# ✅ Good - controlled output
Show me 100 rows from my_large_table

# ⚠️ May return too much data
Show me all rows from my_large_table
```

### 4. Remember Namespace Isolation
- **Personal databases:** Databases prefixed with `u_{username}__`
- **Tenant databases:** Databases for tenants you belong to (e.g., `kbase`, `tenant_name`)

### 5. Leverage Conversation Memory
Claude Code remembers context within a session:
```
> What tables are in my research database?
Claude: You have tables: experiments, results, metrics

> Show me the schema of the first table
Claude: [Shows schema for experiments table]

> Now count how many rows it has
Claude: [Counts rows in experiments table]
```

## Troubleshooting

### Connection Failed

```bash
# Check MCP server health
claude mcp list

# If disconnected, try re-adding
claude mcp remove berdl-datalake
claude mcp add --transport http --scope user berdl-datalake \
  https://hub.berdl.kbase.us/apis/mcp/mcp \
  --header "Authorization: Bearer YOUR_TOKEN"
```

### Authentication Errors

Your token may have expired or be invalid. Get a fresh token:

**Option 1: From BERDL JupyterHub**
```python
import os
token = os.environ["KBASE_AUTH_TOKEN"]
print(f"Your token: {token}")
```

**Option 2: From KBase Narrative**
```python
import os
token = os.environ["KB_AUTH_TOKEN"]
print(f"Your token: {token}")
```

Then update your `~/.claude.json` with the new token.

### Permission Denied

Remember namespace isolation - you can access:
- **Personal databases:** Databases matching pattern `u_{username}__*`
- **Tenant databases:** Databases for tenants you belong to (e.g., `kbase`, `tenant_name`)

```
# Check what you can access
What databases am I allowed to access?
```

### No Response or Timeout

This may indicate:
- BERDL services are down
- Network connectivity issues
- Query is taking too long (e.g., full table scan on large dataset)

Try:
1. Simplifying your query
2. Adding explicit LIMIT clauses
3. Checking BERDL service status
4. Contacting BERDL support team

### Tools Not Available

If Claude Code doesn't recognize BERDL data lake queries:

```bash
# Verify MCP server is connected
claude mcp list

# Should show:
# berdl-datalake: https://hub.berdl.kbase.us/apis/mcp/mcp (HTTP) - ✓ Connected
```

## Updating Your Token

If your KBase token expires or changes:

1. **Edit your configuration:**
```bash
nano ~/.claude.json
```

2. **Update the Authorization header:**
```json
{
  "mcpServers": {
    "berdl-datalake": {
      "type": "http",
      "url": "https://hub.berdl.kbase.us/apis/mcp/mcp",
      "headers": {
        "Authorization": "Bearer NEW_TOKEN_HERE"
      }
    }
  }
}
```

3. **Save and exit** (Ctrl+X, Y, Enter in nano)

4. **Restart Claude Code** for changes to take effect

## Privacy and Security

> **⚠️ Important Data Privacy Warning**
>
> When using Claude Code with the BERDL MCP server:
> - Query results are sent to Anthropic's servers for processing
> - **DO NOT query sensitive, proprietary, or confidential data**
> - Only use with public or test datasets


### Using with VSCode/Cursor

Claude Code can also integrate with VSCode and Cursor IDEs. The same MCP server configuration in `~/.claude.json` is used across all these tools.


