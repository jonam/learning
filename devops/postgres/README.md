# Postgres commands

## use psql inside docker container

use

```
psql.sh
```

OR

```
docker exec -it postgres psql -U postgres -d postgres
```

## Show current database

The prompt should show that but you can also type:

```
SELECT current_database();
```

## Show me tables in current database

```
\dt
```

To show more info:

```
\dt+
```

## Show me all databases

```
\list
```

## Show all system tables

```
\dt pg_catalog.*
```

## how to create a database 

```
CREATE DATABASE mydatabase;
```

OR

```
CREATE DATABASE mydatabase OWNER myuser ENCODING 'UTF8' TEMPLATE template0;
```

## how to switch to the newly created database

```
\c mydatabase
```

## How do I know which user owns the database

```
SELECT datname, pg_catalog.pg_get_userbyid(datdba) as owner FROM pg_catalog.pg_database;
```

should display something like:

```
  datname   |  owner   
------------+----------
 postgres   | postgres
 mydatabase | postgres
 template1  | postgres
 template0  | postgres
(4 rows)
```

## Example create table command

```
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    date_of_birth DATE,
    phone_number VARCHAR(15)
);
```

Insert 10,000 rows into above table:

```
DO $$
DECLARE
    i int := 1;
BEGIN
    WHILE i <= 10000 LOOP
        INSERT INTO customers (first_name, last_name, email, date_of_birth, phone_number)
        VALUES (
            'FirstName' || i::text, 
            'LastName' || i::text, 
            'user' || i::text || '@example.com', 
            ('1990-01-01'::date + (i || ' days')::interval)::date, 
            '123-456-' || LPAD(i::text, 4, '0')
        );
        i := i + 1;
    END LOOP;
END $$;
```

## how I execute a command from a file in psql

```
psql -U username -d databasename < script.sql
```

OR

```
psql -U username -d databasename -f script.sql
```

## How to drop table if not exists in SQL

```
DROP TABLE IF EXISTS table_name;
```
