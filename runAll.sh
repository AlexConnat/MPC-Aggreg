
DATASET=$1
NB_CLIENTS=$2

if [ -z $1 ]
then
    echo "Must supply a dataset {mnist|svhn}"
    exit
fi
if [ -z $2 ]
then
    echo "Must supply a number of clients"
    exit
fi

if [ $DATASET == "mnist" ]
then

    echo "[*] Cleaning up mnist250 folder"
    rm DATA/mnist250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/mnist250/; python3 ../votes_compilator.py ../VOTES_MNIST_250 $NB_CLIENTS; cd ../../;

    gnome-terminal --window -- ./tabsToRun.sh $DATASET $NB_CLIENTS

fi

if [ $DATASET == "svhn" ]
then

    echo "[*] Cleaning up svhn250 folder"
    rm DATA/svhn250/*

    echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
    cd DATA/svhn250/; python3 ../votes_compilator.py ../VOTES_SVHN_250 $NB_CLIENTS; cd ../../;

    gnome-terminal --window -- ./tabsToRun.sh $DATASET $NB_CLIENTS

fi
