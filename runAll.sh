if [ -z $1 ]
then
    echo "Must supply a number of clients!"
    exit
fi

gnome-terminal --window -- ./tabsToRun.sh $1
