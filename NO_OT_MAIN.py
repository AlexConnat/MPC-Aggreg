#!/usr/bin/python3

# Standard python imports
import csv
import sys
import time

# Requirements imports
from mpyc.runtime import mpc
import numpy as np

# Imported from utils.py
from utils import vector_add_all, scalar_add_all, argmax


################################################################################


def usage():
    print(f'Usage: {sys.argv[0]} <mnist|svhn> [nb_samples] [...]')
    sys.exit()

if len(sys.argv) < 2:
    usage()

if sys.argv[1] == 'mnist':
    DATASET = 'mnist250'
elif sys.argv[1] == 'svhn':
    DATASET = 'svhn250'
else:
    print(f'Error: Invalid dataset "{sys.argv[1]}".\n')
    usage()

SERVER_ID = 0
TOTAL_NB_TEACHERS = 250

# Numbers from the Papernot 2018 paper
# Total number of samples have been commented out
if DATASET == 'mnist250':
    #NB_SAMPLES = 10000
    NB_SAMPLES = 640
    NB_CLASSES = 10
    THRESHOLD = 200
    SIGMA1 = 150
    SIGMA2 = 40
elif DATASET == 'svhn250':
    #NB_SAMPLES = 26032
    NB_SAMPLES = 8500
    NB_CLASSES = 10
    THRESHOLD = 300
    SIGMA1 = 200
    SIGMA2 = 40

# If specified, reduce the NB_SAMPLES to this one
if len(sys.argv) > 2:
    NB_SAMPLES = int(sys.argv[2])

# This CSV file will hold the results, each label alongside the time it took to output these labels
timestamp = int(time.time())
csv_file = open(f'BENCHMARK/{DATASET}_{len(mpc.parties)-1}c_{NB_SAMPLES}s_{timestamp}.csv', mode='w')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(['Sample ID','Label','Time elapsed']) # Write CSV Header

# This list will hold the computed labels (or -1 if no label has been given)
# We store them as 8-bit integers for storage purposes
LABELS = np.zeros(NB_SAMPLES, dtype=np.int8)

# This numpy file will hold the "LABELS" array in order of sample_id appearance
numpy_filename = f'labels_{DATASET}_{len(mpc.parties)-1}c_{NB_SAMPLES}s_{timestamp}.npy'
# --> Actually saved at the end of the code on line "np.save(LABELS, numpy)"


################################################################################


# Secure type for integers, and for fixed precision numbers
secint = mpc.SecInt()
secfxp = mpc.SecFxp()

# Number of parties joining the computation (M-1 clients + the server)
M = len(mpc.parties)
print('Number of parties:', M)

# This piece of code is currently ran by 'mpc.pid'
my_pid = mpc.pid
print('my pid:', my_pid)

# The server has party ID 0, every other party is a client
# Code logic is different for the server and the clients
IS_SERVER = False
if my_pid == SERVER_ID:
    IS_SERVER = True
    print('You are the server!')


# Each client loads the votes from its .npy file and compute
# its local parameters
if not IS_SERVER:
    np_votes = np.load(f'DATA/{DATASET}/votes_client_{my_pid}.npy') # For this PoC the path is hardcoded here
    # Compute my local sigma1 and sigma2, local fraction of the total
    # variance respecitively SIGMA1 and SIGMA2
    # The sum of all sigma^2 of all parties is equal to SIGMA^2
    nb_teachers = int(sum(np_votes[0]))
    sigma1 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA1 )
    sigma2 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA2 )



###############################################################################
print('\n' + "="*50)
mpc.run(mpc.start())   #### START THE MPC ROUNDS OF COMPUTATION ####
print("="*50,'\n');
###############################################################################

# Reset the timer before the actual computation begins
last_time = time.time()

# Iterate over all samples in the public dataset
for sample_id in range(NB_SAMPLES):

    if not IS_SERVER:
        # Take the sample_id°th votes of the list
        my_votes = list(map(int, np_votes[sample_id]))

        # Convert the votes to secure type
        my_sec_votes = list(map(secfxp, my_votes))

        # Generate this client's noise part
        my_noise = float(np.random.normal(0, sigma1))

        # Convert the noise part to secure type
        my_sec_noise = secfxp(my_noise)

    else:
        # The server is not an input party (it has no votes, and no noise part)
        my_sec_votes = list(map(secfxp, [None]*NB_CLASSES))
        my_sec_noise = secfxp(None)

    # Secret-share the votes from this party with every other party
    all_sec_votes = mpc.input(my_sec_votes, senders=list(range(1,M)))

    # Aggregate the secure votes from all parties
    total_sec_votes = vector_add_all(all_sec_votes)

    # Secret-share the noise part from this party with every other party
    all_sec_noises = mpc.input(my_sec_noise, senders=list(range(1,M)))

    # Aggregate the secure noises from all parties
    total_sec_noise = scalar_add_all(all_sec_noises)

    # Find the secure maximum in the aggregated array of votes
    sec_max = mpc.max(total_sec_votes)

    # Add the aggregated noise to this max value
    noisy_sec_max = sec_max + total_sec_noise

    # Reveal (=recombine the shares) of this noisy maximum
    noisy_max = float(mpc.run(mpc.output(noisy_sec_max)))

    # If it is lower than a threshold, the teachers are not confident enough
    # Then don't label this sample, and skip to the next one
    if noisy_max < THRESHOLD:
        if IS_SERVER:
            print(f'[*] Sample {sample_id}: NULL')
            LABELS[sample_id] = -1

            elapsed = time.time() - last_time
            # print('time elapsed:', elapsed)
            csv_writer.writerow([sample_id, -1, elapsed])
            last_time = time.time()

        continue # Skip to the next sample_id in the for loop

    # If it is greater than the threshold (meaning that the teachers are
    # confident), we should aggregate a noisy version of the votes vector
    # of each party, and take the argmax of this aggregated array as our label
    else:
        if not IS_SERVER:
            # Each client draws a random gaussian noise vector with NB_CLASSE
            # elements and std deviation sigma2
            my_noise_vector = list(map(float, np.random.normal(0, sigma2, NB_CLASSES)))

            # Add this vector to this client's votes to noise them
            my_noisy_votes = np.array(my_votes) + np.array(my_noise_vector)
            my_noisy_votes = list(map(float, my_noisy_votes))

            # Convert this to secure type
            my_sec_noisy_votes = list(map(secfxp, my_noisy_votes))

        else:
            # The server does no not generate a noise vector, and does not have
            # noisy votes
            my_sec_noisy_votes = list(map(secfxp, [None]*NB_CLASSES))


        # Secret-share the noisy votes from this party with every other party
        all_sec_noisy_votes = mpc.input(my_sec_noisy_votes, senders=list(range(1,M)))

        # Aggregate all noisy votes from all parties
        total_sec_noisy_votes = vector_add_all(all_sec_noisy_votes)

        # Compute the secure argmax on this vector of aggregated noisy votes
        sec_argmax = argmax(total_sec_noisy_votes)

        # Our label is the revealed (=recombined) argmax
        label = mpc.run(mpc.output(sec_argmax, receivers=[SERVER_ID]))

        if label and IS_SERVER: # Should be a tautology as the server is the ONLY one receiving the 'label' variable
            label = int(label)
            print(f'[*] Sample {sample_id}: {label}')
            LABELS[sample_id] = label

            elapsed = time.time() - last_time
            # print('time elapsed:', elapsed)
            csv_writer.writerow([sample_id, label, elapsed])
            last_time = time.time()

        continue # To the next sample_id


###############################################################################
print('\n'+'='*50)
mpc.run(mpc.shutdown())    #### END THE MPC ROUNDS OF COMPUTATION ####
print('='*50)
###############################################################################


# At the end of the computation, store these labels to a .npy file
# if IS_SERVER:
#     np.save(f'labels_{DATASET}_{len(mpc.parties)-1}c_{NB_SAMPLES}s_{timestamp}.npy', LABELS)


# Close the CSV file used for benchmark
csv_file.close()
