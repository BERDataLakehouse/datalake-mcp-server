# CDM MCP Server

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

1. Clone the repository:
   ```bash
   git clone https://github.com/kbase/cdm-mcp-server.git
   cd cdm-mcp-server
   ```

2. Create required directories:
   ```bash
   mkdir -p cdr/cdm/jupyter/cdm_shared_workspace
   ```

3. Make scripts executable:
   ```bash
   chmod +x scripts/*.sh
   ```

4. Create Docker network:
   ```bash
   docker network create cdm-jupyterhub-network
   ```

5. Start the services:
   ```bash
   docker-compose up -d --build
   ```

6. Access the services:
   - MCP Server: http://localhost:8000/docs
   - MinIO Console: http://localhost:9003
   - Spark Master UI: http://localhost:8090
   - JupyterHub: http://localhost:4043

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src tests/
```
