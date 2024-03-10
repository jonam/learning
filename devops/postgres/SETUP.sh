#!/bin/bash

# Getting Started

# If you are here for the first time, and have done nothing yet follow the steps

if  ! ./destroy_postgres.sh; then
    echo "Failed to destroy postgres"
    exit 1
fi

if ! ./start_postgres.sh; then
    echo "Failed to start postgres"
    exit 1
fi

if ! ./setup_db.sh; then
    echo "Failed to setup db"
    exit 1
fi

#if [ ! `./psql.sh` ]; then
#    echo "Failed to enter psql"
#    exit 1
#fi

exit 0
