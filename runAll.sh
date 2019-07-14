#!/bin/bash

#for DATASET in "mnist" "svhn"; do
for DATASET in "mnist"; do

#for NB_CLIENTS in 2 3 4 5 10 15 20 25 30 35 40 50 60 70 80 100 125 200 250; do##for NB_CLIENTS in 10 15 25 50 83 125 250; do
#for NB_CLIENTS in 15 20 25 30 35 40 50 60 70 80 100 125 200 250; do
for NB_CLIENTS in 16 17 18 19 20; do

if [ $DATASET == "mnist" ]
then

    echo "[*] Cleaning up mnist250 folder"
    rm DATA/mnist250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/mnist250/; python3 ../votes_compilator.py ../VOTES_MNIST_250 $NB_CLIENTS; cd ../../;

    #gnome-terminal --window -- ./tabsToRun.sh $DATASET $NB_CLIENTS

fi

if [ $DATASET == "svhn" ]
then

    echo "[*] Cleaning up svhn250 folder"
    rm DATA/svhn250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/svhn250/; python3 ../votes_compilator.py ../VOTES_SVHN_250 $NB_CLIENTS; cd ../../;

    #gnome-terminal --window -- ./tabsToRun.sh $DATASET $NB_CLIENTS

fi

#repeat_params=""
#for ((i=1;i<=NB_CLIENTS+1;i++)); do
#    repeat_params+="-P localhost "
#done

#for ((client_id=1;client_id<=NB_CLIENTS;client_id++)); do
#    python3 main.py $DATASET $repeat_params -I $client_id &
#done

# important to run the server at LAST, because he's the one measuring the
# computation time:
#python3 main.py $DATASET $repeat_params -I 0

NB_PARTIES=$((NB_CLIENTS+1))
python3 main.py $DATASET -M $NB_PARTIES   # handy flag "-M" to say "we are only running $NB_PARTIES localhost parties"

done

done
