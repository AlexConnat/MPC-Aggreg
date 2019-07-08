#!/usr/bin/python3

from mpyc.runtime import mpc
import numpy as np
import sys

from utils import vector_add_all, scalar_add_all, argmax

import time
import csv

if len(sys.argv) != 2:
    print(f'Usage: {sys.argv[0]} <mnist|svhn> [...]')
    sys.exit()

if sys.argv[1] == 'mnist':
    DATASET = 'mnist250'
elif sys.argv[1] == 'svhn':
    DATASET = 'svhn250'
else:
    print('Error: Should only specify "mnist" or "svhn" as datasets')
    print('Received:', sys.argv[1])
    sys.exit()

SERVER_ID = 0
TOTAL_NB_TEACHERS = 250
IS_SERVER = False

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


############### FOR BENCHMARKING PURPOSES ##################
#NB_SAMPLES = 1
csv_file = open(f'BENCHMARK/{DATASET}_{NB_SAMPLES}s_{len(mpc.parties)-1}c.csv', mode='w')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
############################################################



# Will hold the computed labels (or -1 if label is not defined)
# -- 8-bit integers type for storage efficiency --
LABELS = np.zeros(NB_SAMPLES, dtype=np.int8) # 8-bit integers for storage efficiency


# Secure type for integers, and for fixed precision numbers
secint = mpc.SecInt()
secfxp = mpc.SecFxp()

# Number of parties joining the computation (M-1 clients + the server)
M = len(mpc.parties)
print('Number of parties:', M)

# This piece of code is currently running by 'mpc.pid'
my_pid = mpc.pid
print('my pid:', my_pid)

if my_pid == SERVER_ID:
    IS_SERVER = True
    print('You are the server!')


# Each client loads the votes from its .npy file and compute
# its local parameters
if my_pid != SERVER_ID:
    np_votes = np.load(f'DATA/{DATASET}/votes_client_{my_pid}.npy')
    # Compute my local sigma1 and sigma2, a local parts of the total
    # variance respecitively SIGMA1 and SIGMA2
    print('my votes (1st sample):', np_votes[0])
    nb_teachers = int(sum(np_votes[0]))
    sigma1 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA1 )
    sigma2 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA2 )
    print('my noisy votes (e.g.):', list( map(float, np_votes[0] + np.random.normal(0, sigma2, NB_CLASSES)) ))


###############################################################################
print('\n' + "="*50)
mpc.run(mpc.start())   #### START THE MPC ROUNDS OF COMPUTATION ####
print("="*50,'\n');
###############################################################################

last_time = time.time()
elapsed = last_time - mpc.start_time
csv_writer.writerow(['Sample ID','Label','Time elapsed'])

# Iterate over all samples in the public dataset
for sample_id in range(NB_SAMPLES):

    if not IS_SERVER:
        # Take the sample_id°th votes of the list
        my_votes = list(map(int, np_votes[sample_id]))

        # Convert the votes to secure type
        my_sec_votes = list(map(secfxp, my_votes))
    else:
        # Server is not an input party (it has no votes)
        my_sec_votes = list(map(secfxp, [None]*NB_CLASSES))

    # Secret-share the votes from this party with every other party
    all_sec_votes = mpc.input(my_sec_votes, senders=list(range(1,M)))

    # Aggregate the secure votes from all parties
    total_sec_votes = vector_add_all(all_sec_votes)
    #print('total_votes:', mpc.run(mpc.output(total_sec_votes)))

    if not IS_SERVER:
        # Generate two different noise values (from the same distribution)
        # Each party will end up using one of the two in an oblivious manner
        noise0 = float(np.random.normal(0, sigma1))
        noise1 = float(np.random.normal(0, sigma1))

        # Convert them to secure type
        sec_noise0, sec_noise1 = secfxp(noise0), secfxp(noise1)

    else:
        # The server does not generate noise values
        sec_noise0, sec_noise1 = secfxp(None), secfxp(None)

    # Secret-share both noise values with every other party
    all_sec_noises0 = mpc.input(sec_noise0, senders=list(range(1,M)))
    all_sec_noises1 = mpc.input(sec_noise1, senders=list(range(1,M)))

    # # ### DEBUG ### convert them back to float
    # conv_noises0 = mpc.run(mpc.output(all_sec_noises0))
    # conv_noises1 = mpc.run(mpc.output(all_sec_noises1))
    #
    # if not IS_SERVER:
    #     conv_noise0 = float(conv_noises0[my_pid-1])
    #     conv_noise1 = float(conv_noises1[my_pid-1])
    #     print(noise0, conv_noise0, noise0 - conv_noise0)
    #     csv_writer.writerow([noise0, conv_noise0, noise0-conv_noise0])
    #     csv_writer.writerow([noise1, conv_noise1, noise1-conv_noise1])

    if IS_SERVER:
        # Generate a selection bit for every client
        selection_bits = list(map(int, np.random.randint(0,2,M-1)))
        sec_selection_bits = list(map(secfxp, selection_bits))
    else:
        # Only the server draw these random selection bits
        sec_selection_bits = list(map(secfxp, [None]*(M-1)))

    sec_selection_bits = mpc.input(sec_selection_bits, senders=[SERVER_ID])[0] # Only one sender  -->  Only one element (list) in the list
    #print('selection bits:', mpc.run(mpc.output(sec_selection_bits)))

    sec_chosen_noises = []
    for client_id, selection_bit_for_client in enumerate(sec_selection_bits):
        # If b is 0, we select noise0 from this client
        # If b is 0, we select noise1 from this client
        sec_chosen_noise = mpc.if_else(selection_bit_for_client,
                        all_sec_noises1[client_id], all_sec_noises0[client_id])
        sec_chosen_noises.append(sec_chosen_noise)

    #print('chosen noises:', mpc.run(mpc.output(sec_chosen_noises)))

    # Aggregate the secure noise values from all parties
    total_sec_noise = scalar_add_all(sec_chosen_noises)
    #print('total noise:', mpc.run(mpc.output(total_sec_noise)))

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
            csv_writer.writerow([sample_id, 'NULL', elapsed])
            last_time = time.time()

        continue # Skip to the next sample_id in the for loop

    # If it is greater than the threshold (meaning that the teachers are
    # confident), we should aggregate a noisy version of the votes vector
    # of each party, and take the argmax of this aggregated array as our label
    else:
        if not IS_SERVER:
            # Generate two vectors of gaussian noise values
            # Again, which noise vector would be used is oblivious to the clients
            noise_vector0 = list(map(float, np.random.normal(0, sigma2, NB_CLASSES)))
            noise_vector1 = list(map(float, np.random.normal(0, sigma2, NB_CLASSES)))

            # Convert them both to secure type
            sec_noise_vector0 = list(map(secfxp, noise_vector0))
            sec_noise_vector1 = list(map(secfxp, noise_vector1))

        else:
            # The server does not generate noise vectors
            sec_noise_vector0 = list(map(secfxp, [None]*NB_CLASSES))
            sec_noise_vector1 = list(map(secfxp, [None]*NB_CLASSES))


        # Secret-share both noise vectors with every other party
        all_sec_noise_vector0 = mpc.input(sec_noise_vector0, senders=list(range(1,M)))
        all_sec_noise_vector1 = mpc.input(sec_noise_vector1, senders=list(range(1,M)))

        sec_chosen_noise_vectors = []
        for client_id, selection_bit_for_client in enumerate(sec_selection_bits):
            # If b is 0, we select noise0 from this client
            # If b is 0, we select noise1 from this client
            # FIXME: Condition must be integral!
            selection_bit_for_client.integral = True
            sec_chosen_noise_vector = mpc.if_else(selection_bit_for_client, all_sec_noise_vector1[client_id], all_sec_noise_vector0[client_id])
            sec_chosen_noise_vectors.append(sec_chosen_noise_vector)
            #sec_chosen_noise_vectors.append(all_sec_noise_vector1[client_id])

        #print('chosen noise vectors:', mpc.run(mpc.output(sec_chosen_noise_vectors)))


        # Aggregate the secure noise vectors from all parties
        total_sec_noise = vector_add_all(sec_chosen_noise_vectors)
        #print('total noise vector:', mpc.run(mpc.output(total_sec_noise)))

        # Add this secure total noise vector to the secure total votes vector
        total_sec_noisy_votes = mpc.vector_add(total_sec_votes, total_sec_noise)

        # Compute the secure argmax on this vector of aggregated noisy votes
        sec_argmax = argmax(total_sec_noisy_votes)

        # Our label is the revealed (=recombined) argmax
        label = int(mpc.run(mpc.output(sec_argmax)))

        if IS_SERVER:
            print(f'[*] Sample {sample_id}: {label}')
            LABELS[sample_id] = label

            elapsed = time.time() - last_time
            # print('time elapsed:', elapsed)
            csv_writer.writerow([sample_id, label, elapsed])
            last_time = time.time()

        continue # To the next sample_id


if IS_SERVER:
    pass
    # At the end of the computation, store these labels to a .npy file
    # Also store the computation time in the filename

    #from datetime import datetime
    #ts = int(datetime.timestamp(datetime.now()))
    # import time
    # elapsed = time.time() - mpc.start_time
    #
    # np.save(f'./RESULTS/labels_{DATASET}_{M-1}_clients_{elapsed}.npy', LABELS)



############### FOR BENCHMARKING PURPOSES ##################
csv_file.close()
############################################################


###############################################################################
print('\n'+'='*50)
mpc.run(mpc.shutdown())    #### END THE MPC ROUNDS OF COMPUTATION ####
print('='*50)
###############################################################################


# TODO: Care about performance? SecInt() SecFxp(), use less bits? use server to
# select b? Skip first round of communication? (total number of teachers?)
# Do fancy optimizations to the code?
