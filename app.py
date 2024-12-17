from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi.responses import FileResponse
import mysql.connector
import psycopg2
import os
import tempfile
import logging
import io
import random
import string
import pandas as pd
from typing import Dict, Any
import aiohttp
from datetime import datetime
import json
import re
from io import StringIO
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from dotenv import load_dotenv
import aiomysql
import asyncpg
from typing import Union

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Add this line to reduce multipart logging
logging.getLogger("multipart.multipart").setLevel(logging.INFO)

# Add these debug logs right before load_dotenv()
logger.debug(f"Current working directory: {os.getcwd()}")
logger.debug(f"Looking for .env file in: {os.path.join(os.getcwd(), '.env')}")
logger.debug(f".env file exists: {os.path.exists(os.path.join(os.getcwd(), '.env'))}")

# Then load the env vars
load_dotenv()

# After loading, let's also check all environment variables
logger.debug("All environment variables:")
logger.debug(f"Environment vars: {dict(os.environ)}")  # This will show all env vars except passwords

# Database type mappings
type_mapping = {
    'int64': 'INTEGER',
    'float64': 'NUMERIC',
    'bool': 'BOOLEAN',
    'datetime64[ns]': 'TIMESTAMP',
    'object': 'TEXT'
}

# Add this import at the top with other imports
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Add this class before app initialization
class LargeUploadMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/upload-file"):
            # Increase the body limit for upload endpoints
           request._body_size_limit = 1.5 * 1024 * 1024 * 1024  # 1.5GB
        return await call_next(request)

# Initialize FastAPI and add middlewares in correct order
app = FastAPI()

# First add the upload size middleware
app.add_middleware(LargeUploadMiddleware)

# Simplify CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load environment variables
aws_database_name = os.getenv("AWS_DATABASE_NAME")
aws_host = os.getenv("AWS_HOST")
aws_user = os.getenv("AWS_USER")
aws_password = os.getenv("AWS_PASSWORD")

azure_database_name = os.getenv("AZURE_DATABASE_NAME")
azure_host = os.getenv("AZURE_HOST")
azure_user = os.getenv("AZURE_USER")
azure_password = os.getenv("AZURE_PASSWORD")

neon_host = os.getenv("NEON_HOST")
neon_database = os.getenv("NEON_DATABASE")
neon_user = os.getenv("NEON_USER")
neon_password = os.getenv("NEON_PASSWORD")

filessio_host = os.getenv("FILESSIO_HOST")
filessio_database = os.getenv("FILESSIO_DATABASE")
filessio_user = os.getenv("FILESSIO_USER")
filessio_password = os.getenv("FILESSIO_PASSWORD")
filessio_port = int(os.getenv("FILESSIO_PORT", 3307))

FLOWISE_API_ENDPOINT = os.getenv("FLOWISE_API_ENDPOINT")

# Add these environment variables with the others after load_dotenv()
aiven_database_name = os.getenv("AIVEN_DATABASE_NAME")
aiven_host = os.getenv("AIVEN_HOST")
aiven_user = os.getenv("AIVEN_USER")
aiven_password = os.getenv("AIVEN_PASSWORD")
aiven_port = int(os.getenv("AIVEN_PORT", 23490))

# Add these environment variables with the others after load_dotenv()
aiven_database_name_mysql = os.getenv("AIVEN_DATABASE_NAME_MYSQL")
aiven_host_mysql = os.getenv("AIVEN_HOST_MYSQL")
aiven_user_mysql = os.getenv("AIVEN_USER_MYSQL")
aiven_password_mysql = os.getenv("AIVEN_PASSWORD_MYSQL")
aiven_port_mysql = int(os.getenv("AIVEN_PORT_MYSQL", 23490))

# Add these debug logs
logger.debug("Aiven Environment Variables:")
logger.debug(f"AIVEN_DATABASE_NAME: {aiven_database_name}")
logger.debug(f"AIVEN_HOST: {aiven_host}")
logger.debug(f"AIVEN_USER: {aiven_user}")
logger.debug(f"AIVEN_PORT: {aiven_port}")
# Not logging password for security reasons

def create_aws_connection():
    connection = mysql.connector.connect(
        host=aws_host,
        user=aws_user,
        password=aws_password,
        database=aws_database_name
    )
    return connection

def create_azure_connection():
    connection = mysql.connector.connect(
        host=azure_host,
        user=azure_user,
        password=azure_password,
        database=azure_database_name,
        ssl_disabled=True
    )
    return connection

def create_neon_connection():
    connection = psycopg2.connect(
        host=neon_host,
        database=neon_database,
        user=neon_user,
        password=neon_password,
        sslmode='require'
    )
    return connection

def create_filessio_connection():
    connection = mysql.connector.connect(
        host=filessio_host,
        user=filessio_user,
        password=filessio_password,
        database=filessio_database,
        port=filessio_port
    )
    return connection

    

def create_aiven_connection():
    connection = psycopg2.connect(
        host=aiven_host,
        user=aiven_user,
        password=aiven_password,
        database=aiven_database_name,
        port=aiven_port
    )
    return connection

def create_aivenmysql_connection():
    connection = mysql.connector.connect(
        host=aiven_host_mysql,
        user=aiven_user_mysql,
        password=aiven_password_mysql,
        database=aiven_database_name_mysql,
        port=aiven_port_mysql
    )
    return connection

def get_connection(cloud: str):
    if cloud == "azure":
        return create_azure_connection()
    elif cloud == "aws":
        return create_aws_connection()
    elif cloud == "neon":
        return create_neon_connection()
    elif cloud == "filessio":
        return create_filessio_connection()
    elif cloud == "aivenmysql":
        return create_aivenmysql_connection()
    elif cloud == "aiven":
        return create_aiven_connection()
    else:
        raise ValueError("Invalid cloud provider specified. Valid options are: 'azure', 'aws', 'neon', 'filessio', 'aiven', 'aivenmysql'")

async def create_async_connection(cloud: str):
    """Creates async database connections based on cloud provider."""
    try:
        if cloud in ["aws", "azure", "filessio", "aivenmysql"]:
            # MySQL connections
            if cloud == "aivenmysql":
                # Special case for aivenmysql
                connection = await aiomysql.connect(
                    host=aiven_host_mysql,
                    user=aiven_user_mysql,
                    password=aiven_password_mysql,
                    db=aiven_database_name_mysql,
                    port=aiven_port_mysql,
                    autocommit=True
                )
            else:
                # Other MySQL providers
                connection = await aiomysql.connect(
                    host=globals()[f"{cloud}_host"],
                    user=globals()[f"{cloud}_user"],
                    password=globals()[f"{cloud}_password"],
                    db=globals()[f"{cloud}_database"],
                    port=globals().get(f"{cloud}_port", 3306),
                    autocommit=True
                )
            return connection, "mysql"
            
        elif cloud in ["neon", "aiven"]:
            # PostgreSQL connections
            if cloud == "aiven":
                # Special case for aiven
                connection = await asyncpg.connect(
                    host=aiven_host,
                    user=aiven_user,
                    password=aiven_password,
                    database=aiven_database_name,
                    port=aiven_port
                )
            else:
                # Neon connection
                connection = await asyncpg.connect(
                    host=neon_host,
                    user=neon_user,
                    password=neon_password,
                    database=neon_database,
                    port=globals().get(f"{cloud}_port", 5432)
                )
            return connection, "postgresql"
            
        else:
            raise ValueError(f"Unsupported cloud provider: {cloud}")
            
    except Exception as e:
        logger.error(f"Error creating async connection for {cloud}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to database: {str(e)}"
        )

@app.get("/sqlquery/")
async def sqlquery(sqlquery: str, cloud: str, request: Request):
    """Execute SQL query using async database connections."""
    logger.debug(f"Received API call: {request.url} with cloud parameter: {cloud}")
    
    connection = None
    try:
        # Create async connection
        connection, db_type = await create_async_connection(cloud)
        
        # Execute query based on database type
        if db_type == "mysql":
            async with connection.cursor() as cursor:
                await cursor.execute(sqlquery)
                
                if cursor.description:  # SELECT query
                    headers = [i[0] for i in cursor.description]
                    results = await cursor.fetchall()
                else:  # Non-SELECT query
                    await connection.commit()
                    return {"status": "Query executed successfully"}
                    
        else:  # PostgreSQL
            if sqlquery.strip().lower().startswith('select'):
                results = await connection.fetch(sqlquery)
                headers = [k for k in results[0].keys()] if results else []
            else:
                await connection.execute(sqlquery)
                return {"status": "Query executed successfully"}
        
        # Create temporary file for results
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt") as temp_file:
            if db_type == "mysql":
                temp_file.write(" | ".join(headers) + "\n")
                for row in results:
                    temp_file.write(" | ".join(str(item) for item in row) + "\n")
            else:
                temp_file.write(" | ".join(headers) + "\n")
                for row in results:
                    temp_file.write(" | ".join(str(row[k]) for k in headers) + "\n")
            temp_file_path = temp_file.name
            
        logger.debug(f"Query executed successfully, results written to {temp_file_path}")
        return FileResponse(path=temp_file_path, filename="output.txt", media_type='text/plain')
        
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if connection:
            if isinstance(connection, aiomysql.Connection):
                connection.close()
            else:
                await connection.close()
            logger.debug("Database connection closed")

@app.middleware("http")
async def remove_temp_file(request, call_next):
    logger.debug(f"Processing request: {request.url}")
    response = await call_next(request)
    if isinstance(response, FileResponse) and os.path.exists(response.path):
        try:
            os.remove(response.path)
            logger.debug(f"Temporary file {response.path} removed successfully")
        except Exception as e:
            logger.error(f"Error removing temp file: {e}")
    return response

def generate_table_name(base_name: str) -> str:
    """Generates a sanitized and unique table name based on base_name."""
    clean_base = ''.join(c for c in base_name if c.isalnum() or c == '_')
    clean_base = clean_base.upper()[:30]  # Convert to uppercase instead of lowercase
    random_suffix = ''.join(random.choices(string.digits, k=5))
    table_name = f"{clean_base}_{random_suffix}"
    logger.debug(f"Generated table name: {table_name}")
    return table_name

@app.post("/upload-file-llm-pg/")
async def upload_file_llm_pg(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Handles file uploads using LLM for schema detection and creates PostgreSQL table."""
    logger.info(f"Starting LLM-assisted file upload process for file: {file.filename}")
    
    try:
        # Read file in chunks and analyze first chunk for schema
        CHUNK_SIZE = 1024 * 1024  # 1MB chunks
        first_chunk = await file.read(CHUNK_SIZE)
        
        # Ensure proper string decoding with error handling
        try:
            sample_data = first_chunk.decode('utf-8').split('\n')[:5]
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            sample_data = first_chunk.decode('latin-1').split('\n')[:5]
        
        # Detect delimiter from first line
        first_line = sample_data[0]
        delimiter = ',' if ',' in first_line else '|'
        logger.info(f"Detected delimiter: '{delimiter}'")
        
        # Reset file pointer for later reading
        await file.seek(0)
        
        # Convert sample data to string for LLM prompt
        sample_data_str = '\n'.join(sample_data)
        
        # Prepare prompt for LLM
        prompt = {
            "question": f"""You are a PostgreSQL schema generator. Your task is to analyze the data and output ONLY a JSON schema.
            DO NOT include any explanations, markdown formatting, or additional text.
            ONLY output the JSON schema in the exact format specified below.
            
            Data sample (delimiter: '{delimiter}'):
            {sample_data_str}
            
            Required output format (exact JSON structure):
            {{
                "columns": [
                    {{"name": "column_name", "type": "postgresql_type", "description": "brief description"}}
                ]
            }}
            
            Rules:
            1. Use standard PostgreSQL types (TEXT, NUMERIC, DATE, TIMESTAMP. Dont use INTEGER)
            2. Column names must be SQL-safe (alphanumeric and underscores only)
            3. Use NUMERIC for all values that can have decimal points, even if some rows contain whole numbers. INTEGER not be used.
            4. Use DATE or TIMESTAMP for dates
            5. Use TEXT when unsure
            
            Output only the JSON schema, nothing else.
            """
        }

        # Call LLM API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                FLOWISE_API_ENDPOINT,
                json=prompt,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM API error: {await response.text()}"
                    )
                schema_suggestion = await response.json()
                logger.info(f"Received schema suggestion from LLM: {schema_suggestion}")

        # Parse the LLM response
        try:
            response_text = schema_suggestion.get('text', '')
            parsed_schema = json.loads(response_text)
            
            if 'columns' not in parsed_schema:
                raise ValueError("Schema does not contain 'columns' key")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM schema: {str(e)}"
            )

        # Generate table name and create table
        table_name = generate_table_name(os.path.splitext(file.filename)[0])
        logger.info(f"Generated table name: {table_name}")
        
        # Create Neon connection with error handling
        try:
            connection = create_neon_connection()
            cursor = connection.cursor()
            
            # Create table using LLM-suggested schema
            columns = []
            for col in parsed_schema['columns']:
                col_name = ''.join(c for c in col['name'] if c.isalnum() or c == '_')
                columns.append(f'"{col_name}" {col["type"]}')
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS public.{table_name} (
                    {', '.join(columns)}
                )
            """
            cursor.execute(create_table_query)
            logger.info("Table created successfully with LLM-suggested schema")

            # Create a temporary file for efficient COPY
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='', delete=False) as temp_file:
                # Read and write file content in chunks
                while True:
                    chunk = await file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    try:
                        chunk_str = chunk.decode('utf-8')
                    except UnicodeDecodeError:
                        chunk_str = chunk.decode('latin-1')
                    temp_file.write(chunk_str)
                
                temp_file_path = temp_file.name

            # Use COPY command with the temporary file
            start_time = datetime.now()
            
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                cursor.copy_expert(
                    f"""
                    COPY public.{table_name}
                    FROM STDIN
                    WITH (FORMAT CSV, DELIMITER '{delimiter}', HEADER true)
                    """,
                    f
                )
            
            connection.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get final count
            cursor.execute(f"SELECT COUNT(*) FROM public.{table_name}")
            final_count = cursor.fetchone()[0]
            
            logger.info(f"Bulk insert completed: {final_count} rows in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "message": f"File uploaded and {final_count} rows inserted in {duration:.2f} seconds",
                "table_name": table_name,
                "rows_inserted": final_count,
                "duration_seconds": duration,
                "rows_per_second": final_count / duration if duration > 0 else 0,
                "columns": parsed_schema['columns'],
                "llm_schema_used": True
            }

        except Exception as e:
            logger.error(f"Database operation error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Database operation failed: {str(e)}"
            )
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
            # Clean up temporary file
            if 'temp_file_path' in locals():
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error removing temp file: {str(e)}")
            logger.info("Database connection closed and temporary files cleaned up")
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/upload-file-llm-mysql/")
async def upload_file_llm_mysql(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """Handles file uploads using LLM for schema detection and creates MySQL table."""
    logger.info(f"Starting LLM-assisted MySQL file upload process for file: {file.filename}")
    
    try:
        # Read file contents
        contents = await file.read()
        logger.info(f"Successfully read file contents, size: {len(contents)} bytes")

        # Detect delimiter
        first_line = contents.decode('utf-8').split('\n')[0]
        delimiter = ',' if ',' in first_line else '|'
        logger.info(f"Detected delimiter: '{delimiter}'")

        # Get first 5 lines for schema analysis
        lines = contents.decode('utf-8').split('\n')[:5]
        sample_data = '\n'.join(lines)
        
        # Prepare prompt for LLM - Modified for MySQL
        prompt = {
            "question": f"""You are a MySQL schema generator. Your task is to analyze the data and output ONLY a JSON schema.
            DO NOT include any explanations, markdown formatting, or additional text.
            ONLY output the JSON schema in the exact format specified below.
            
            Data sample (delimiter: '{delimiter}'):
            {sample_data}
            
            Required output format (exact JSON structure):
            {{
                "columns": [
                    {{"name": "column_name", "type": "mysql_type", "description": "brief description"}}
                ]
            }}
            
            Rules:
            1. Use standard MySQL types (VARCHAR(255), DECIMAL(20,6), DATE, TIMESTAMP. Dont use INT)
            2. Column names must be SQL-safe (alphanumeric and underscores only)
            3. Use DECIMAL(20,6) for numeric values that can have decimal points, even if some rows contain whole numbers. INT not be used.
            5. Use VARCHAR(255) for text fields
            6. Use DATE for dates
            7. Use TIMESTAMP for datetime fields
            8. When in doubt, use VARCHAR(255)
            
            Output only the JSON schema, nothing else.
            """
        }

        # Call LLM API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                FLOWISE_API_ENDPOINT,
                json=prompt,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM API error: {await response.text()}"
                    )
                schema_suggestion = await response.json()
                logger.info(f"Received schema suggestion from LLM: {schema_suggestion}")

        # Extract JSON from the LLM response
        try:
            response_text = schema_suggestion.get('text', '')
            parsed_schema = json.loads(response_text)
            
            logger.info(f"Successfully parsed schema from LLM response: {parsed_schema}")
            
            if 'columns' not in parsed_schema:
                raise ValueError("Schema does not contain 'columns' key")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.error(f"Raw LLM response: {schema_suggestion}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM schema: {str(e)}"
            )

        try:
            df = pd.read_csv(
                io.StringIO(contents.decode('utf-8')), 
                delimiter=delimiter
            )
            logger.info(f"Successfully created DataFrame: {len(df)} rows, {len(df.columns)} columns")

            # Generate table name
            table_name = generate_table_name(os.path.splitext(file.filename)[0])
            logger.info(f"Generated table name: {table_name}")
            
            # Connect to MySQL (FILESSIO)
            connection = mysql.connector.connect(
                host=filessio_host,
                user=filessio_user,
                password=filessio_password,
                database=filessio_database,
                port=filessio_port,
                allow_local_infile=True,
                charset='utf8mb4',
                use_pure=False,
                pool_size=32,
                max_allowed_packet=1073741824  # 1GB
            )
            cursor = connection.cursor()
            logger.info("Successfully connected to MySQL database")

            # Add these performance settings right after creating the cursor
            cursor.execute("SET foreign_key_checks = 0")
            cursor.execute("SET unique_checks = 0")
            cursor.execute("SET autocommit = 0")
            cursor.execute("SET sql_log_bin = 0")  # Disable binary logging if possible
            
            # Create table using LLM-suggested schema
            columns = []
            for col in parsed_schema['columns']:
                col_name = ''.join(c for c in col['name'] if c.isalnum() or c == '_')
                columns.append(f"`{col_name}` {col['type']}")
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    {', '.join(columns)}
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(create_table_query)
            logger.info("Table created successfully with LLM-suggested schema")

            # Modify the batch insert section
            batch_size = 25000  # Increased from 1000
            total_batches = (len(df) + batch_size - 1) // batch_size
            rows_inserted = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]
                values = [tuple(row) for _, row in batch.iterrows()]
                placeholders = ', '.join(['%s'] * len(df.columns))
                insert_query = f"""
                    INSERT INTO `{table_name}` 
                    VALUES ({placeholders})
                """
                cursor.executemany(insert_query, values)
                
                # Commit every 10 batches or at the end
                if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(df):
                    connection.commit()
                
                rows_inserted += len(values)
                logger.info(f"Progress: inserted batch {(i//batch_size)+1}/{total_batches} ({rows_inserted}/{len(df)} rows)")

            # Reset the performance settings before closing
            cursor.execute("SET foreign_key_checks = 1")
            cursor.execute("SET unique_checks = 1")
            cursor.execute("SET autocommit = 1")
            cursor.execute("SET sql_log_bin = 1")  # Re-enable binary logging if it was disabled

            logger.info(f"All data inserted successfully into {table_name}")

            # Verify row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            final_count = cursor.fetchone()[0]
            logger.info(f"Final row count in table: {final_count}")

            return {
                "status": "success",
                "message": "File uploaded and data inserted successfully",
                "table_name": table_name,
                "rows_inserted": rows_inserted,
                "final_row_count": final_count,
                "columns": parsed_schema['columns'],
                "llm_schema_used": True,
                "connection_details": {
                    "host": filessio_host,
                    "database": filessio_database,
                    "user": filessio_user
                }
            }

        except Exception as e:
            if 'connection' in locals():
                connection.rollback()
            logger.error(f"Error processing data: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process data: {str(e)}"
            )
        finally:
            if 'connection' in locals():
                cursor.close()
                connection.close()
                logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        logger.exception("Full traceback:")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/connect-db/")
async def connect_db(
    request: Request,
    host: str,
    database: str,
    user: str,
    password: str,
    port: int = 3306,
    db_type: str = "mysql",
    sqlquery: str = None,
) -> Any:
    """
    Execute SQL query using provided database connection credentials.
    """
    logger.debug(f"Received API call: {request.url}")
    logger.info(f"Attempting to connect to {db_type} database at {host}")

    connection = None
    try:
        # Create connection based on database type
        if db_type.lower() == "mysql":
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
        elif db_type.lower() == "postgresql":
            connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid database type. Supported types are 'mysql' and 'postgresql'"
            )

        logger.info("Database connection established successfully")

        # If no query provided, just test connection and return success
        if not sqlquery:
            if connection:
                connection.close()
            return {
                "status": "success",
                "message": "Database connection successful",
                "connection_details": {
                    "host": host,
                    "database": database,
                    "user": user,
                    "db_type": db_type,
                    "port": port
                }
            }

        # Execute query
        try:
            cursor = connection.cursor()
            cursor.execute(sqlquery)
            
            # Handle queries that return results
            if cursor.description is not None:
                headers = [i[0] for i in cursor.description]
                results = cursor.fetchall()
                cursor.close()

                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=".txt") as temp_file:
                    temp_file.write(" | ".join(headers) + "\n")
                    for row in results:
                        temp_file.write(" | ".join(str(item) for item in row) + "\n")
                    temp_file_path = temp_file.name

                logger.debug(f"Query executed successfully, results written to {temp_file_path}")
                # Return the file response
                response = FileResponse(path=temp_file_path, filename="output.txt", media_type='text/plain')
                return response
            
            # Handle non-SELECT queries
            else:
                connection.commit()
                cursor.close()
                logger.debug("Non-SELECT query executed successfully")
                return {"status": "Query executed successfully"}

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Query execution failed: {str(e)}"
            )
        finally:
            if 'cursor' in locals():
                cursor.close()

    except (mysql.connector.Error, psycopg2.Error) as e:
        logger.error(f"Database connection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database connection failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    finally:
        if connection:
            try:
                connection.close()
                logger.debug("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")

@app.post("/upload-file-custom-db-pg/")
async def upload_file_custom_db_pg(
    host: str,
    database: str,
    user: str,
    password: str,
    file: UploadFile = File(...),
    schema: str = "public",
    port: int = 5432
) -> Dict[str, Any]:
    """Handles file uploads using LLM for schema detection and creates PostgreSQL table in a custom database."""
    logger.info(f"Starting LLM-assisted file upload process for file: {file.filename} to custom PostgreSQL database")
    
    try:
        # Read file contents
        contents = await file.read()
        buffer = StringIO(contents.decode('utf-8'))
        logger.info(f"Successfully read file contents, size: {len(contents)} bytes")

        # Detect delimiter
        first_line = contents.decode('utf-8').split('\n')[0]
        delimiter = ',' if ',' in first_line else '|'
        logger.info(f"Detected delimiter: '{delimiter}'")

        # Get first 5 lines for schema analysis
        lines = contents.decode('utf-8').split('\n')[:5]
        sample_data = '\n'.join(lines)
        
        # Prepare prompt for LLM
        prompt = {
            "question": f"""You are a PostgreSQL schema generator. Your task is to analyze the data and output ONLY a JSON schema.
            DO NOT include any explanations, markdown formatting, or additional text.
            ONLY output the JSON schema in the exact format specified below.
            
            Data sample (delimiter: '{delimiter}'):
            {sample_data}
            
            Required output format (exact JSON structure):
            {{
                "columns": [
                    {{"name": "column_name", "type": "postgresql_type", "description": "brief description"}}
                ]
            }}
            
            Rules:
            1. Use standard PostgreSQL types (TEXT, NUMERIC, DATE, TIMESTAMP. Dont use INTEGER)
            2. Column names must be SQL-safe (alphanumeric and underscores only)
            3. Use NUMERIC for all values that can have decimal points, even if some rows contain whole numbers. INTEGER not be used.
            4. Use DATE or TIMESTAMP for dates
            5. Use TEXT when unsure
            
            Output only the JSON schema, nothing else.
            """
        }

        # Call LLM API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                FLOWISE_API_ENDPOINT,
                json=prompt,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM API error: {await response.text()}"
                    )
                schema_suggestion = await response.json()
                logger.info(f"Received schema suggestion from LLM: {schema_suggestion}")

        # Parse the LLM response
        try:
            response_text = schema_suggestion.get('text', '')
            parsed_schema = json.loads(response_text)
            
            if 'columns' not in parsed_schema:
                raise ValueError("Schema does not contain 'columns' key")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM schema: {str(e)}"
            )

        # Generate table name
        table_name = generate_table_name(os.path.splitext(file.filename)[0])
        logger.info(f"Generated table name: {table_name}")
        
        # Connect to custom PostgreSQL database
        try:
            connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
                options='-c statement_timeout=900000'  # 1 hour
            )
            cursor = connection.cursor()
            
            # Create schema if it doesn't exist
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            
            # Create table using LLM-suggested schema
            columns = []
            for col in parsed_schema['columns']:
                col_name = ''.join(c for c in col['name'] if c.isalnum() or c == '_')
                columns.append(f'"{col_name}" {col["type"]}')
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                    {', '.join(columns)}
                )
            """
            cursor.execute(create_table_query)
            logger.info("Table created successfully with LLM-suggested schema")

            # Use COPY command for bulk insert
            buffer.seek(0)  # Reset buffer to start
            start_time = datetime.now()
            
            cursor.copy_expert(
                f"""
                COPY {schema}.{table_name}
                FROM STDIN
                WITH (FORMAT CSV, DELIMITER '{delimiter}', HEADER true)
                """,
                buffer
            )
            
            connection.commit()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get final count
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table_name}")
            final_count = cursor.fetchone()[0]
            
            logger.info(f"Bulk insert completed: {final_count} rows in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "message": f"File uploaded and {final_count} rows inserted in {duration:.2f} seconds",
                "table_name": f"{schema}.{table_name}",
                "rows_inserted": final_count,
                "duration_seconds": duration,
                "rows_per_second": final_count / duration if duration > 0 else 0,
                "columns": parsed_schema['columns'],
                "llm_schema_used": True,
                "connection_details": {
                    "host": host,
                    "database": database,
                    "schema": schema,
                    "user": user
                }
            }

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
            logger.info("Database connection closed")
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

@app.post("/upload-file-custom-db-mysql/")
async def upload_file_custom_db_mysql(
    host: str,
    database: str,
    user: str,
    password: str,
    file: UploadFile = File(...),
    port: int = 3306,
    sslmode: str = None
) -> Dict[str, Any]:
    """Handles file uploads using LLM for schema detection and creates MySQL table in a custom database."""
    logger.info(f"Starting LLM-assisted file upload process for file: {file.filename} to custom MySQL database")
    
    try:
        # Read file contents
        contents = await file.read()
        buffer = StringIO(contents.decode('utf-8'))
        logger.info(f"Successfully read file contents, size: {len(contents)} bytes")

        # Detect delimiter
        first_line = contents.decode('utf-8').split('\n')[0]
        delimiter = ',' if ',' in first_line else '|'
        logger.info(f"Detected delimiter: '{delimiter}'")

        # Get first 5 lines for schema analysis
        lines = contents.decode('utf-8').split('\n')[:5]
        sample_data = '\n'.join(lines)
        
        # Prepare prompt for LLM
        prompt = {
            "question": f"""You are a MySQL schema generator. Your task is to analyze the data and output ONLY a JSON schema.
            DO NOT include any explanations, markdown formatting, or additional text.
            ONLY output the JSON schema in the exact format specified below.
            
            Data sample (delimiter: '{delimiter}'):
            {sample_data}
            
            Required output format (exact JSON structure):
            {{
                "columns": [
                    {{"name": "column_name", "type": "mysql_type", "description": "brief description"}}
                ]
            }}
            
            1. Use standard MySQL types (VARCHAR(255), DECIMAL(20,6), DATE, TIMESTAMP. Dont use INT)
            2. Column names must be SQL-safe (alphanumeric and underscores only)
            3. Use DECIMAL(20,6) for numeric values that can have decimal points, even if some rows contain whole numbers. INT not be used.
            5. Use VARCHAR(255) for text fields
            6. Use DATE for dates
            7. Use TIMESTAMP for datetime fields
            8. When in doubt, use VARCHAR(255)
            
            Output only the JSON schema, nothing else.
            """
        }

        # Call LLM API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                FLOWISE_API_ENDPOINT,
                json=prompt,
                headers={"Content-Type": "application/json"}
            ) as response:
                if not response.ok:
                    raise HTTPException(
                        status_code=500,
                        detail=f"LLM API error: {await response.text()}"
                    )
                schema_suggestion = await response.json()
                logger.info(f"Received schema suggestion from LLM: {schema_suggestion}")

        # Parse the LLM response
        try:
            response_text = schema_suggestion.get('text', '')
            parsed_schema = json.loads(response_text)
            
            if 'columns' not in parsed_schema:
                raise ValueError("Schema does not contain 'columns' key")
                
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse LLM schema: {str(e)}"
            )

        # Generate table name
        table_name = generate_table_name(os.path.splitext(file.filename)[0])
        logger.info(f"Generated table name: {table_name}")
        
        try:
            # Connect to custom MySQL database with SSL if specified
            connection_params = {
                "host": host,
                "database": database,
                "user": user,
                "password": password,
                "port": port,
                "allow_local_infile": True
            }
            
            # Add SSL if specified
            if sslmode and sslmode.lower() == 'require':
                connection_params.update({
                    "ssl_verify_cert": True,
                    "ssl_verify_identity": True
                })
                
            connection = mysql.connector.connect(**connection_params)
            cursor = connection.cursor()
            
            # Create table using LLM-suggested schema
            columns = []
            for col in parsed_schema['columns']:
                col_name = ''.join(c for c in col['name'] if c.isalnum() or c == '_')
                columns.append(f"`{col_name}` {col['type']}")
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS `{table_name}` (
                    {', '.join(columns)}
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            cursor.execute(create_table_query)
            logger.info("Table created successfully with LLM-suggested schema")

            # Instead of LOAD DATA LOCAL INFILE, use batch insert since some MySQL servers restrict LOAD DATA
            start_time = datetime.now()
            
            df = pd.read_csv(
                buffer,
                delimiter=delimiter,
                quoting=3  # QUOTE_NONE
            )
            
            # Batch insert
            batch_size = 15000
            total_rows = len(df)
            total_batches = (total_rows + batch_size - 1) // batch_size
            rows_inserted = 0
            
            cursor.execute("SET foreign_key_checks = 0")
            cursor.execute("SET unique_checks = 0")
            cursor.execute("SET autocommit = 0")
            
            try:
                placeholders = ', '.join(['%s'] * len(df.columns))
                insert_query = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
                
                for i in range(0, total_rows, batch_size):
                    batch = df.iloc[i:min(i + batch_size, total_rows)]
                    values = list(map(tuple, batch.values))
                    cursor.executemany(insert_query, values)
                    
                    rows_inserted += len(values)
                    
                    # Add commit logging
                    if (i // batch_size) % 10 == 0 or (i + batch_size) >= total_rows:
                        connection.commit()
                        logger.info(f"Committed rows to database. Total rows committed so far: {rows_inserted:,}")
                    
                    logger.info(f"Progress: inserted batch {(i//batch_size)+1}/{total_batches} ({rows_inserted:,}/{total_rows:,} rows)")
            
            finally:
                # Final commit with logging
                if rows_inserted % (batch_size * 10) != 0:
                    connection.commit()
                    logger.info(f"Final commit completed. Total rows committed: {rows_inserted:,}")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get final count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            final_count = cursor.fetchone()[0]
            
            logger.info(f"Bulk insert completed: {final_count} rows in {duration:.2f} seconds")
            
            return {
                "status": "success",
                "message": f"File uploaded and {final_count} rows inserted in {duration:.2f} seconds",
                "table_name": table_name,
                "rows_inserted": final_count,
                "duration_seconds": duration,
                "rows_per_second": final_count / duration if duration > 0 else 0,
                "columns": parsed_schema['columns'],
                "llm_schema_used": True,
                "connection_details": {
                    "host": host,
                    "database": database,
                    "user": user
                }
            }

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()
            logger.info("Database connection closed")
            
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout=900)

