#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mms;
    CREATE USER mms WITH PASSWORD 'mmspassword';
    GRANT ALL PRIVILEGES ON DATABASE mms TO mms;
    \c mms
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    GRANT ALL ON SCHEMA public TO mms;
EOSQL
