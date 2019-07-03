
NB_CLIENTS=$1

if [ -z $1 ]
then
    echo "Must supply a number of clients!"
    exit
fi

echo "[*] Cleaning up mnist250 folder"
rm DATA/mnist250/*

echo "[*] Running the votes compilator for $NB_CLIENTS clients (equal split)"
cd DATA/mnist250/; python3 ../votes_compilator.py ../VOTES_MNIST_250 $NB_CLIENTS; cd ../../;

gnome-terminal --window -- ./tabsToRun.sh $NB_CLIENTS
