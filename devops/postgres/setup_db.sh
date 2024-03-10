docker exec -it postgres /bin/bash -c "psql -U postgres -d postgres < /workspace/manoj/github/jonam/devops/postgres/setup_database.sql"
docker exec -it postgres /bin/bash -c "psql -U postgres -d mydatabase < /workspace/manoj/github/jonam/devops/postgres/setup_customers.sql"
