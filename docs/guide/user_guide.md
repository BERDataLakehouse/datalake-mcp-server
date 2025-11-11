# BERDL Datalake MCP Server User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Creating Sample Delta Tables](#creating-sample-delta-tables)
4. [Using the API](#using-the-api)
5. [AI Assistant Integration](#ai-assistant-integration)
   - [MCP Configuration](#mcp-configuration)
   - [MCP Host Setup](#mcp-host-setup)
   - [Example Prompts](#example-prompts)

## Introduction

The BERDL Datalake MCP Server is a FastAPI-based service that enables AI assistants to interact with Delta Lake tables stored in MinIO through Spark. It implements the Model Context Protocol (MCP) to provide LLM-accessible tools for data operations.

> **⚠️ Important Warning:** 
> 
> This service allows arbitrary `read-oriented` queries to be executed against Delta Lake tables. Query results will be sent to the model host server, unless you are hosting your model locally.

> **❌** Additionally, this service is **NOT** approved for deployment to any production environment, including CI, until explicit approval is granted by KBase leadership. Use strictly for local development or evaluation purposes only.

### Key Features
- List and explore Delta Lake tables and databases
- Read table data and schemas
- Execute SQL queries with safety constraints
- Secure access control with KBase authentication
- Integration with AI assistants through MCP

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

## Creating Sample Delta Tables

### Option 1: Using Python Script

You can create sample Delta tables directly using PySpark. Create a Python script with the following code:
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import random

# Create Spark session (adjust configuration as needed)
spark = SparkSession.builder \
    .appName("CreateSampleTables") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Create test namespace
namespace = 'test'
spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")

# Create employees data
schema = StructType([
    StructField("id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("department", StringType(), True),
    StructField("salary", IntegerType(), True)
])

departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
data = []
for i in range(1, 101):
    data.append((
        i, 
        f"Employee {i}", 
        departments[random.randint(0, len(departments)-1)], 
        random.randint(50000, 150000)
    ))

employees_df = spark.createDataFrame(data, schema)
employees_df.show(5)

# Save employees table to S3 and Hive metastore
(
employees_df.write
    .mode("overwrite")
    .option("path", "s3a://cdm-lake/employees")
    .format("delta")
    .saveAsTable(f"{namespace}.employees")
)

# Create products data
product_schema = StructType([
    StructField("product_id", IntegerType(), False),
    StructField("name", StringType(), True),
    StructField("category", StringType(), True),
    StructField("price", IntegerType(), True),
    StructField("in_stock", IntegerType(), True)
])

categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
products = []
for i in range(1, 51):
    products.append((
        i, 
        f"Product {i}", 
        categories[random.randint(0, len(categories)-1)], 
        random.randint(10, 500),
        random.randint(0, 100)
    ))

product_df = spark.createDataFrame(products, product_schema)
product_df.show(5)

# Save products table to S3 and Hive metastore
(
product_df.write
    .mode("overwrite")
    .option("path", "s3a://cdm-lake/products")
    .format("delta")
    .saveAsTable(f"{namespace}.products")
)
```
### Option 2: Check and Create Bucket Manually

1. Open MinIO Console at `http://localhost:9003`
2. Login with user `minio` and password `minio123`
3. Create a new bucket named `cdm-lake`
4. Use this bucket for your Delta tables

## Using the API

#### List Databases
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/databases/list' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"use_hms": true}'
```

#### List Tables
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/databases/tables/list' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"database": "default", "use_hms": true}'
```

#### Get Table Schema
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/databases/tables/schema' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"database": "default", "table": "products"}'
```

#### Get Sample Data
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/tables/sample' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"database": "default", "table": "products", "limit": 10}'
```

#### Count Table Rows
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/tables/count' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"database": "default", "table": "products"}'
```

#### Query Table
```bash
curl -X 'POST' \
  'http://localhost:8000/apis/mcp/delta/tables/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer your-kbase-token-here' \
  -d '{"query": "SELECT * FROM test.products WHERE price > 100 LIMIT 5"}'
```

## AI Assistant Integration

The BERDL Datalake MCP Server implements the Model Context Protocol (MCP), allowing AI assistants to interact with your Delta Lake tables using natural language. This section explains how to configure and use the server with various MCP-compatible clients.

> **⚠️ Reminder** 
> 
> Any data returned by AI generated query will be sent to the AI model host server (e.g. ChatGPT, Anthropic, etc.)—unless you're running the model locally. Only create or import test datasets that are safe to share publicly when using this service.  


### MCP Configuration

1. Create or update your MCP configuration file at `~/.mcp/mcp.json`:
   ```json
   {
     "mcpServers": {
       "delta-lake-mcp": {
         "url": "http://localhost:8000/apis/mcp/mcp",
         "enabled": true,
         "headers": {
           "Authorization": "Bearer YOUR_KBASE_TOKEN"
         }
       }
     }
   }
   ```

2. Replace `YOUR_KBASE_TOKEN` with your actual KBase authentication token.

3. Run `chmod 600` on the `mcp.json` file to prevent it from being read by unauthorized parties/tools.

### MCP Host Setup
Most MCP‑enabled tools offer two ways to configure a host:

- **Direct JSON import**  
  Look for an "Import MCP config" or "Load from file" option and point it to your `~/.mcp/mcp.json`.

- **Manual entry**  
  Open the MCP or API settings, then copy the URL, headers, and other fields from your `mcp.json` into the corresponding inputs.  


#### Cherry Studio

1. Open Cherry Studio settings
2. Go to "Integrations" → "MCP"
3. Click "Add Server"
4. Configure with:
   - Name: delta-lake-mcp
   - URL: http://localhost:8000/apis/mcp/mcp
   - Headers:
     ```json
     {
       "Authorization": "Bearer YOUR_KBASE_TOKEN"
     }
     ```
5. Save and restart Cherry Studio

#### Cursor

1. Open Cursor settings (⌘,)
2. Navigate to "Model Context Protocol" in the sidebar
3. Click "Add MCP Server"
4. Enter the following details:
   - Name: delta-lake-mcp
   - URL: http://localhost:8000/apis/mcp/mcp
   - Headers:
     ```json
     {
       "Authorization": "Bearer YOUR_KBASE_TOKEN"
     }
     ```
5. Click "Save" and restart Cursor

### Example Prompts

Once configured, you can interact with your Delta tables using natural language. Here are some example prompts:

> **⚠️ Final Warning Before We Phone Home!**  
> 
> Any output from an AI generated query will be sent to the remote AI model host—unless you've configured your MCP Host to use a locally deployed model.

#### Database Exploration
```markdown
1. "List all databases in the system"
2. "Show me the tables in the test database"
3. "What's the schema of the products table?"
```

#### Data Analysis
```markdown
1. "Count the total number of employees"
2. "Show me 5 random employees from the Engineering department"
3. "What's the average salary by department?"
4. "List all products that are out of stock"
5. "Find the top 3 most expensive products in each category"
```

#### Complex Queries
```markdown
1. "Compare the salary distribution between Engineering and Marketing departments"
2. "Find products with price above average for their category"
3. "Show me employees who earn more than their department's average salary"
```

#### Tips for Effective Prompts
- Be specific about the data you want to see
- Mention the table names if you're working with multiple tables
- Specify any filters or conditions clearly
- Use natural language to describe aggregations or calculations

## Using the BERDL Datalake MCP Server Deployed in Rancher2
Use the steps below to access the BERDL Datalake MCP Server deployed in Rancher 2 Kubernetes cluster from your local machine.

> **⚠️ Warning**  
> 
> AI query results are sent to a remote host unless using a local model. Ensure your data is public and you have permission from the original author to use it.

### Prerequisites
- KBase CI auth token
- SSH access to login1.berkeley.kbase.us

### Step 1: Set Up SSH Tunnel

Add the following line to your `/etc/hosts` file:
```
127.0.0.1 cdmhub.ci.kbase.us
```

Create an SSH tunnel with port forwarding to access the internal KBase network:

```bash
# Create SSH tunnel with port forwarding
sudo ssh -fN \
  -L 443:cdmhub.ci.kbase.us:443 \
  <ac.anl_username>@login1.berkeley.kbase.us
```

Replace `<ac.anl_username>` with your actual ANL username.

For more information on SSH tunnels, refer to the [BERDL JupyterHub User Guide](https://github.com/BERDataLakehouse/BERDL_JupyterHub/blob/main/docs/user_guide.md#1-create-ssh-tunnel).

### Step 2: Update MCP Configuration
Create or update your MCP configuration file at `~/.mcp/mcp.json`:

> **🔑 Authentication Note**
> The currently deployed BERDL Datalake MCP Server is configured to use **CI KBase auth server**. Please ensure you are using your **CI KBase Auth token** (not production tokens) for authorization.

```json
{
  "mcpServers": {
    "delta-lake-mcp": {
      "url": "https://cdmhub.ci.kbase.us/apis/mcp/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer YOUR_CI_KBASE_AUTH_TOKEN"
      }
    }
  }
}
```

### Step 3: Cherry Studio SSL Certificate Fix

When using Cherry Studio with the SSH tunnel, you may encounter an SSL certificate error:

```
Start failed
Error invoking remote method 'mcp:list-tools': Error:
[MCP] Error activating server CDM MCP Server: SSE error: TypeError: fetch failed: unable to verify the first certificate
```

This happens because the SSH tunnel creates a localhost connection, but the SSL certificate is issued for `cdmhub.ci.kbase.us`, causing certificate verification to fail.

**Solution for Cherry Studio (macOS):**

Launch Cherry Studio with SSL verification disabled:

```bash
NODE_TLS_REJECT_UNAUTHORIZED=0 /Applications/Cherry\ Studio.app/Contents/MacOS/Cherry\ Studio
```
`NODE_TLS_REJECT_UNAUTHORIZED=0` disables Node.js SSL certificate verification. This allows the connection to proceed despite certificate validation issues

**Note:** VS Code/Cursor typically handle certificates gracefully and don't require this workaround.
