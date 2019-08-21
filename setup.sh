bold=$(tput bold)
normal=$(tput sgr0)

echo "${bold}[*] Updating package source lists...${normal}"
sudo apt update

echo ""
echo "${bold}[*] Installing python3 and pip3${normal}"
sudo apt install python3 python3-pip -y

echo ""
echo "${bold}[*] Installing libgmp, libmpfr, and libmpc for gmpy2${normal}"
sudo apt install libgmp-dev libmpfr-dev libmpc-dev -y

echo ""
echo "${bold}[*] Installing numpy, gmpy2, and mpyc${normal}"
pip3 install numpy gmpy2 git+https://github.com/lschoe/mpyc
