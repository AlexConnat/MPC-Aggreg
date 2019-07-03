#!/usr/bin/python3

from mpyc.runtime import mpc
import numpy as np

from utils import vector_add_all, scalar_add_all, argmax


SERVER_ID = 0
DATASET = 'mnist250'
TOTAL_NB_TEACHERS = 250
IS_SERVER = False

if DATASET == 'mnist250':
    NB_SAMPLES = 10000
    NB_CLASSES = 10
    THRESHOLD = 200
    SIGMA1 = 150
    SIGMA2 = 40
elif DATASET == 'svhn250':
    NB_SAMPLES = 26032
    NB_CLASSES = 10
    THRESHOLD = 300
    SIGMA1 = 200
    SIGMA2 = 40


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
    nb_teachers = int(sum(np_votes[0]))
    sigma1 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA1 )
    sigma2 = float( np.sqrt(float(nb_teachers)/float(TOTAL_NB_TEACHERS)) * SIGMA2 )


###############################################################################
print('\n' + "="*50)
mpc.run(mpc.start())   #### START THE MPC ROUNDS OF COMPUTATION ####
print("="*50,'\n'); 
###############################################################################



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
        noise0 = float(np.random.normal(0,SIGMA1))
        noise1 = float(np.random.normal(0,SIGMA1))
        #print('noise0:', noise0)
        #print('noise1:', noise1)

        # Convert them to secure type
        sec_noise0, sec_noise1 = secfxp(noise0), secfxp(noise1)
    else:
        # The server does not generate noise values
        sec_noise0, sec_noise1 = secfxp(None), secfxp(None)

    # Secret-share both noise values with every other party
    all_sec_noises0 = mpc.input(sec_noise0, senders=list(range(1,M)))
    all_sec_noises1 = mpc.input(sec_noise1, senders=list(range(1,M)))

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
    noisy_max = float(mpc.run(mpc.output(noisy_sec_max))) # TODO: only server receives?

    # If it is lower than a threshold, the teachers are not confident enough
    # Then don't label this sample, and skip to the next one
    if noisy_max < THRESHOLD:
        if IS_SERVER:
            print(f'[*] Sample {sample_id}: NULL')
        continue # Skip to the next sample_id in the for loop

    # If it is greater than the threshold (meaning that the teachers are
    # confident), we should aggregate a noisy version of the votes vector
    # of each party, and take the argmax of this aggregated array as our label
    else:
        if not IS_SERVER:
            # Generate the vector of gaussian noise values
            noise_vector = np.random.normal(0, SIGMA2, NB_CLASSES)
    
            # Add it to the votes to noise them
            noisy_votes = list(map(float, np.array(my_votes) + noise_vector))

            # Convert them to secure type
            sec_noisy_votes = list(map(secfxp, noisy_votes))
        
        else:
            # The server does not have input votes
            sec_noisy_votes = list(map(secfxp, [None]*NB_CLASSES))

        # Secret-share them with every party
        all_sec_noisy_votes = mpc.input(sec_noisy_votes, senders=list(range(1,M)))

        # Aggregate all these noisy votes from every party
        total_sec_noisy_votes = vector_add_all(all_sec_noisy_votes)

        # Compute the secure argmax on this aggregated vector
        sec_argmax = argmax(total_sec_noisy_votes)
    
        # Our label is the revealed (=recombined) argmax
        label = int(mpc.run(mpc.output(sec_argmax))) # TODO: Only server receives label??
        
        if IS_SERVER:
            print(f'[*] Sample {sample_id}: {label}')
        
        continue # To the next sample_id

###############################################################################
print('\n'+'='*50)
mpc.run(mpc.shutdown())
print('='*50)
###############################################################################


# TODO: Care about performance? SecInt() SecFxp(), use less bits? use server to
# select b? Skip first round of communication? (total number of teachers?)
# Do fancy optimizations to the code?
