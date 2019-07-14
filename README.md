# MPC-Aggreg



### Requirements

To install the `gmpy2` on Linux with `pip`, you must already have installed libgmp, libmpfr and libmpc (Tested on Ubuntu 18.04).

`sudo apt install libgmp-dev libmpfr-dev libmpc-dev`



Then install the only 3 python requirements, numpy, gmpy2 and mpyc.

`pip3 install numpy gmpy2 git+https://github.com/lschoe/mpyc`



### Notes

The code is in `main.py` and it only calls one external file called `utils.py` containing some helper functions.

For the purpose of the PoC, this code requires to be ran in this directory structure - namely having a directory `DATA/` containing at least `mnist250/` and `svhn250/` - and having a (potentially empty) directory `BENCHMARK/`.



**MPC-Aggreg/** <br>
| .gitignore <br>
| main.py <br>
| utils.py <br>
| <br>
| **DATA/** <br>
|  |  *mnist250/* <br>
|  |  |  votes_client_i.npy <br>
|   |  |  […] <br>
|  |  *svhn250/* <br>
|  |  |  votes_client_i.npy <br>
|  |  |  […] <br>
| <br>
| **BENCHMARK/** <br>
|  |  mnist250_640s_5c.csv <br>
|  |  […] <br>
| <br>
| runAll.sh <br>



To generate these votes `votes_client_i.npy`, where `i` is the cliend ID, we wrote a little script `votes_compilator.py` that you find in the `DATA/`directory (more information in the README file of the `DATA/` directory).



The script `runAll.sh` was used to generate the CSV files used in the benchmarking of this PoC. <br>
What it does, is running the code `main.py` with both datasets "mnist" and "svhn" and different number of clients ranging from 2 (3-party MPC) to 250 (251-party MPC).
