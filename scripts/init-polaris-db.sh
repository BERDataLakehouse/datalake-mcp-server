#!/bin/bash
set -e

# Create the polaris database in the shared PostgreSQL instance.
# This script is mounted into /docker-entrypoint-initdb.d/ and runs once
# when the PostgreSQL container is first initialized.

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE polaris'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'polaris')\gexec
EOSQL

echo "Polaris database created (or already exists)"
