# 🚀 FastAPI Server for Connecting REX-C to a Fixed Database

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

This FastAPI server provides a simple way to connect LLMs, AI tools, or frontends to any database. It includes multiple endpoints tailored for various use cases. For detailed explanations of each endpoint, visit the 'Build' section at [rex.tigzig.com](https://rex.tigzig.com).

For REX-C, we would be using the sqlquery endpoint to connect to a fixed database.

---

### 🛠️ Build Command

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

---

### 🚀 Run Command

To start the server, execute:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

---

### 🔑 Environment Variables and Database Connections
REX-C only needs the fixed database connection that you would be using. In case of the demo, it is linked to the aiven connection. You can use any of the existing connections or setup a new one in similar format.

Keep in mind to to use the right connector depending upon the database you are using i.e. MySQL or PostgreSQL.

---

## 📡 Endpoints

This section provides a quick overview of the available endpoints, what they do, and the parameters they require. These endpoints allow connecting, uploading, and querying databases using FastAPI.

---

#### 1. **SQL Query Execution Endpoint**
**`GET /sqlquery/`**  
- **Description**: Executes a SQL query on a specified database. Supports `SELECT` and `non-SELECT` queries. Results from `SELECT` queries are returned as a text file.  
- **Parameters**:  
  - `sqlquery` (string): The SQL query to execute.  
  - `cloud` (string): The database provider (`azure`, `aws`, `neon`, `filessio`,'aiven').  
- **Authentication**: Uses credentials from environment variables for the specified database provider. Useful if want to provide options across a set of hardcoded databases. 

This is the only endpoint that REX-C would be using.

---

## 🔒 CORS Configuration

By default, the server accepts requests from all origins (`*`). If you want to restrict access to specific domains, you can modify the CORS middleware configuration in `app.py`. Here's an example of how to whitelist specific domains:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tigzig.com",
        "https://*.tigzig.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
```

This configuration will only allow requests from `tigzig.com` and its subdomains. Adjust the `allow_origins` list according to your security requirements.


