-- dangerous command
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    date_of_birth DATE,
    phone_number VARCHAR(15)
);

DO $$
DECLARE
    i int := 1;
BEGIN
    WHILE i <= 100000 LOOP
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
END $$
