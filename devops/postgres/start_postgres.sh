container_name=postgres
if ! docker run --name $container_name -p 5432:5432 -v $HOME/assign:/workspace -e POSTGRES_PASSWORD=popat -d postgres; then
    echo "Could not spawn docker"
    exit 1
fi

# Wait for the container to be in a running state
until [ "$(docker inspect -f {{.State.Running}} $container_name)" == "true" ]; do
    echo "Waiting for docker to be running"
    sleep 1
done

sleep 15
