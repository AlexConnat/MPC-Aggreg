sudo apt update
sudo apt install python3 python3-pip -y
sudo apt install libgmp-dev libmpfr-dev libmpc-dev -y
pip3 install numpy gmpy2 git+https://github.com/lschoe/mpyc
git clone https://github.com/AlexConnat/MPC-Aggreg
