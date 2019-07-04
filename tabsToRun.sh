#!/bin/bash

DATASET=$1
NB_CLIENTS=$2

repeat_params=""
for ((i=1;i<=NB_CLIENTS+1;i++)); do    
    repeat_params+="-P localhost "
done

for ((client_id=1;client_id<=NB_CLIENTS;client_id++)); do    
    cmd="python3 main.py $DATASET $repeat_params -I $client_id"
    gnome-terminal --tab -t "Client $client_id" -- bash -c "$cmd"
done

# important to run the server at LAST, because he's the one measuring the
# computation time:
cmd="python3 main.py $DATASET $repeat_params -I 0"
gnome-terminal --tab -t "Server" -- bash -c "$cmd; exec bash" # The server's terminal will stay open
