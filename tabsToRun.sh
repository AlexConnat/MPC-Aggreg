#!/bin/bash

NB_CLIENTS=$1

repeat_params=""
for ((i=1;i<=NB_CLIENTS+1;i++)); do    
    repeat_params+="-P localhost "
done

cmd="python3 server_all_main.py $repeat_params -I 0"
gnome-terminal --tab -t "Server" -- bash -c "$cmd"

for ((client_id=1;client_id<=NB_CLIENTS;client_id++)); do    
    cmd="python3 server_all_main.py $repeat_params -I $client_id"
    gnome-terminal --tab -t "Client $client_id" -- bash -c "$cmd"
done
