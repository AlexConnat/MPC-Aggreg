#!/usr/bin/python3

from mpyc.runtime import mpc
import numpy as np

from utils import vector_add_all, scalar_add_all, argmax

# Secure type for integers, and for fixed precision numbers
secint = mpc.SecInt()
secfxp = mpc.SecFxp()

# Number of parties joining the computation
M = len(mpc.parties)
print('Number of parties:', M)

# This piece of code is currently running by 'mpc.pid'
my_pid = mpc.pid
print('my pid:', my_pid)

if my_pid == 4:
    print('YOU ARE THE SERVER!') #TODO: For the rest, not nice to branch on PIDs

# Load the votes from .npy file
np_votes = np.load(f'DATA/mnist250/votes_client_{my_pid}.npy')

# Derive these parameters from file
NB_SAMPLES = len(np_votes)
NB_CLASSES = len(np_votes[0])
nb_teachers = int(sum(np_votes[0]))

# Some useful constants TODO: SHOULD SMARTLY DEPEND ON THE TOTAL NUMBER OF TEACHERS?
SIGMA1 = 150
SIGMA2 = 40     
THRESHOLD = 200 


###############################################################################
print('\n' + "="*50)
mpc.run(mpc.start())   #### START THE MPC ROUNDS OF COMPUTATION ####
print("="*50,'\n'); 
###############################################################################


# Compute the total number of teachers (by adding every party local
# number of teachers
all_nb_teachers = mpc.input(secfxp(nb_teachers))
sec_total_nb_teachers = scalar_add_all(all_nb_teachers)
total_nb_teachers = mpc.run(mpc.output(sec_total_nb_teachers))

# Compute my local sigma1 and sigma2, a local parts of the total
# variance respecitively SIGMA1 and SIGMA2
sigma1 = float( np.sqrt(float(nb_teachers)/float(total_nb_teachers)) * SIGMA1 )
sigma2 = float( np.sqrt(float(nb_teachers)/float(total_nb_teachers)) * SIGMA2 )


# Iterate over all samples in the public dataset
for sample_id in range(1):
    
    # Take the sample_id°th votes of the list
    my_votes = list(map(int, np_votes[sample_id]))

    # Convert the votes to secure type
    my_sec_votes = list(map(secfxp, my_votes))

    # Secret-share the votes from this party with every other party 
    all_sec_votes = mpc.input(my_sec_votes)

    # Aggregate the secure votes from all parties
    total_sec_votes = vector_add_all(all_sec_votes)

    # Generate two different noise values (from the same distribution)
    # Each party will end up using one of the two in an oblivious manner # TODO: Server explanation blabla??
    noise0 = float(np.random.normal(0,SIGMA1))
    noise1 = float(np.random.normal(0,SIGMA1))

    # Convert them to secure type
    sec_noise0, sec_noise1 = secfxp(noise0), secfxp(noise1)

    # Secret-share both noise values with every other party
    all_sec_noises0 = mpc.input(sec_noise0)
    all_sec_noises1 = mpc.input(sec_noise1)
    print('all_sec_noises0 =', mpc.run(mpc.output(all_sec_noises0)))
    print('all_sec_noises1 =', mpc.run(mpc.output(all_sec_noises1)))


    ###################################################################################
    # Generate a random selection bit
    b = mpc.random_bit(secfxp)# FIXME: Will be the SAME random bit for every party
    print('b =', mpc.run(mpc.output(b)))

    # This will choose to keep noise0 or noise1 depending on the value of b
    # b is not known by the party, this is an hence an oblivious choice
    # TODO: Server should choose which to take from all_sec_noises0 and which to take from all_sec_noises1
    chosen_sec_noises = mpc.if_else(b, all_sec_noises1, all_sec_noises0)
    ###################################################################################

    # Aggregate the secure noise values from all parties 
    total_sec_noise = scalar_add_all(chosen_sec_noises)

    # Find the secure maximum in the aggregated array of votes
    sec_max = mpc.max(total_sec_votes)

    # Add the aggregated noise to this max value
    noisy_sec_max = sec_max + total_sec_noise

    # Reveal (=recombine the shares) of this noisy maximum 
    noisy_max = float(mpc.run(mpc.output(noisy_sec_max, receivers=range(0,M))))

    # If it is lower than a threshold, the teachers are not confident enough
    # Then don't label this sample, and skip to the next one
    if noisy_max < THRESHOLD:
        print(f'[*] Sample {sample_id}: NULL')
        continue # Skip to the next sample_id in the for loop

    # If it is greater than the threshold (meaning that the teachers are
    # confident), we should aggregate a noisy version of the votes vector
    # of each party, and take the argmax of this aggregated array as our label
    else:
        # Generate the vector of gaussian noise values
        noise_vector = np.random.normal(0, SIGMA2, NB_CLASSES)
    
        # Add it to the votes to noise them
        noisy_votes = list(map(float, np.array(my_votes) + noise_vector))

        # Convert them to secure type
        sec_noisy_votes = list(map(secfxp, noisy_votes))

        # Secret-share them with every party
        all_sec_noisy_votes = mpc.input(sec_noisy_votes)

        # Aggregate all these noisy votes from every party
        total_sec_noisy_votes = vector_add_all(all_sec_noisy_votes)

        # Compute the secure argmax on this aggregated vector
        sec_argmax = argmax(total_sec_noisy_votes)
    
        # Our label is the revealed (=recombined) argmax
        label = int(mpc.run(mpc.output(sec_argmax)))

        print(f'[*] Sample {sample_id}: {label}')
        continue

###############################################################################
print('\n'+'='*50)
mpc.run(mpc.shutdown())
print('='*50)
###############################################################################


# TODO: Care about performance? SecInt() SecFxp(), use less bits? use server to
# select b? Skip first round of communication? (total number of teachers?)
# Do fancy optimizations to the code?
