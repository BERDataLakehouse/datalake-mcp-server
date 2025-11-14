# BERDL Datalake MCP Server

A FastAPI-based service that enables AI assistants to interact with Delta Lake tables stored in MinIO through Spark, implementing the Model Context Protocol (MCP) for natural language data operations.

> **⚠️ Important Warning:** 
> 
> This service allows arbitrary `read-oriented` queries to be executed against Delta Lake tables. Query results will be sent to the model host server, unless you are hosting your model locally.

> **❌** Additionally, this service is **NOT** approved for deployment to any production environment, including CI, until explicit approval is granted by KBase leadership. Use strictly for local development or evaluation purposes only.

## Documentation

For detailed documentation, please refer to the [User Guide](docs/guide/user_guide.md). The guide covers:

- [Quick Start](docs/guide/user_guide.md#quick-start) - Bring the local service up and running
- [Creating Sample Delta Tables](docs/guide/user_guide.md#creating-sample-delta-tables) - Set up local test data
- [Using the API](docs/guide/user_guide.md#using-the-api) - Direct API usage examples
- [AI Assistant Integration](docs/guide/user_guide.md#ai-assistant-integration) - Configure and use with MCP Host tools
  - [MCP Configuration](docs/guide/user_guide.md#mcp-configuration) - Create `mcp.json`
  - [MCP Host Setup](docs/guide/user_guide.md#mcp-host-setup) - Configure MCP Host
  - [Example Prompts](docs/guide/user_guide.md#example-prompts) - Natural language examples

## Quick Start

### Option A: Using Pre-Built Images (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/BERDataLakehouse/datalake-mcp-server.git
   cd datalake-mcp-server
   ```

2. Edit `docker-compose.yaml`:
   - Uncomment all `image:` and `platform:` lines
   - Comment out all `build:` sections

3. Start the services:
   ```bash
   docker compose up -d
   ```

### Option B: Building from Source (For Developers)

1. Clone required repositories:
   ```bash
   # Clone at the same directory level as datalake-mcp-server
   cd ..
   git clone https://github.com/BERDataLakehouse/spark_notebook_base.git
   git clone https://github.com/BERDataLakehouse/kube_spark_manager_image.git
   git clone https://github.com/BERDataLakehouse/hive_metastore.git
   cd datalake-mcp-server
   ```

2. Build base images:
   ```bash
   cd ../spark_notebook_base
   docker build -t spark_notebook_base:local .
   cd ../datalake-mcp-server
   ```

3. Ensure `docker-compose.yaml` has `build:` sections uncommented (default)

4. Start the services:
   ```bash
   docker compose up -d --build
   ```

### Access the Services

- **MCP Server API**: http://localhost:8000/apis/mcp/docs
- **MCP Server Root**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9003 (credentials: minio/minio123)
- **Spark Master UI**: http://localhost:8090
- **PostgreSQL**: localhost:5432 (credentials: hive/hivepassword)

**Note**: The MCP server is mounted at `/apis/mcp` by default. Set `SERVICE_ROOT_PATH=""` environment variable to serve at root.

## Authentication

The service uses KBase authentication with role-based access control:

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KBASE_AUTH_URL` | No | `https://ci.kbase.us/services/auth/` | KBase authentication service URL |
| `KBASE_ADMIN_ROLES` | No | `KBASE_ADMIN` | Comma-separated list of KBase roles with full admin access |
| `KBASE_REQUIRED_ROLES` | No | `BERDL_USER` | Comma-separated list of KBase roles required to authenticate. Users must have all these roles |

### Testing

```bash
# Install dependencies (only required on first run or when the uv.lock file changes)
uv sync --locked

# Run tests
PYTHONPATH=. uv run pytest tests

# Run with coverage
PYTHONPATH=. uv run pytest --cov=src tests/
```
