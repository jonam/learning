if [ ! `docker stop postgres` ]; then
    echo "Container postgres probably not running."
fi

if [ ! `docker rm postgres` ]; then
    echo "Container postgres does not exist."
fi

exit 0
