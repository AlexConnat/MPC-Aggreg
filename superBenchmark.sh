#!/bin/bash

ifname="lo"
limit="100Gbps"

for LATENCY in 0 0.5 5 25 50 100 200; do

# Remove all network limitations
sudo tc qdisc del dev $ifname root 

# Apply network rate and latency limitations
sudo tc qdisc add dev $ifname root handle 1: htb default 12
sudo tc class add dev $ifname parent 1:1 classid 1:12 htb rate $limit ceil $limit
sudo tc qdisc add dev $ifname parent 1:12 netem delay $LATENCY

echo "---- LATENCY = $LATENCY ms ----"

for DATASET in "mnist"; do

for NB_CLIENTS in {2..10}; do

if [ $DATASET == "mnist" ]
then
    echo "[*] Cleaning up mnist250 folder"
    rm DATA/mnist250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/mnist250/; python3 ../votes_compilator.py ../VOTES_MNIST_250 $NB_CLIENTS; cd ../../;
fi

if [ $DATASET == "svhn" ]
then
    echo "[*] Cleaning up svhn250 folder"
    rm DATA/svhn250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/svhn250/; python3 ../votes_compilator.py ../VOTES_SVHN_250 $NB_CLIENTS; cd ../../;
fi

NB_PARTIES=$((NB_CLIENTS+1))
python3 main.py $DATASET -M $NB_PARTIES   # handy flag "-M" to say "we are only running $NB_PARTIES localhost parties"

echo ""
echo ""

done

done

done
