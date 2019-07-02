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

# Load the votes from .npy file
np_votes = np.load(f'svhn250/votes_client_{my_pid}.npy')

# Derive these parameters from file
NB_SAMPLES = len(np_votes)
NB_CLASSES = len(np_votes[0])
nb_teachers = int(sum(np_votes[0]))

# Some useful constants
SIGMA1 = 200
SIGMA2 = 40     
THRESHOLD = 300 


###############################################################################
print('\n' + "="*50); mpc.run(mpc.start()); print("="*50,'\n'); 
###############################################################################



# TODO: TODO TODO TODO List on them all!!!
SAMPLE_ID = 156

my_votes = list(map(int, np_votes[SAMPLE_ID]))
print('my votes:', my_votes, '\n')



# COMPUTE MY PARAMS
all_nb_teachers = mpc.input(secfxp(nb_teachers))
sec_total_nb_teachers = scalar_add_all(all_nb_teachers)
total_nb_teachers = mpc.run(mpc.output(sec_total_nb_teachers))
print(f'my ratio of teachers: {nb_teachers}/{total_nb_teachers}', )

sigma1 = float( np.sqrt(float(nb_teachers)/float(total_nb_teachers)) * SIGMA1 )
print(f'sigma1: {sigma1}  (/{SIGMA1})')

sigma2 = float( np.sqrt(float(nb_teachers)/float(total_nb_teachers)) * SIGMA2 )
print(f'sigma2: {sigma2}  (/{SIGMA2})')


print()


# Convert the votes to secure type
my_sec_votes = list(map(secfxp, my_votes))

# Secret-share the votes from this party with every other party 
all_sec_votes = mpc.input(my_sec_votes)

# Aggregate the secure votes from all parties
total_sec_votes = vector_add_all(all_sec_votes)
print('total votes:', mpc.run(mpc.output(total_sec_votes)))
print()

# Generate two different noise values (from the same distribution)
# Each party will end up using one of the two in an oblivious manner # TODO: Server explanation blabla??
noise0 = float(np.random.normal(0,SIGMA1))
noise1 = float(np.random.normal(0,SIGMA1))
print('noise0 =', noise0)
print('noise1 =', noise1)

print()

# Convert them to secure type
sec_noise0, sec_noise1 = secfxp(noise0), secfxp(noise1)

# Secret-share both noise values with every other party
all_sec_noises0 = mpc.input(sec_noise0)
all_sec_noises1 = mpc.input(sec_noise1)


###################################################################################
# TODO: Server side maaaaagic!
# Selection bit
b = mpc.random_bit(secfxp)# FIXME: Will be the SAME random bit for every party
print('selection bit:', mpc.run(mpc.output(b)))

# Will choose to keep noise0 or noise1 depending on the value of b
# TODO: Server should choose which to take from all_sec_noises0 and which to take from all_sec_noises1
chosen_sec_noises = mpc.if_else(b, all_sec_noises1, all_sec_noises0)
print('chosen_sec_noises:', mpc.run(mpc.output(chosen_sec_noises)))
###################################################################################

# Aggregate the secure noise values from all parties 
total_sec_noise = scalar_add_all(chosen_sec_noises)
print('total noise:', mpc.run(mpc.output(total_sec_noise)))
print()

# Find the secure maximum in the aggregated array of votes
sec_max = mpc.max(total_sec_votes)
print('maximum:', mpc.run(mpc.output(sec_max)))

# Add the aggregated noise to this max value
noisy_sec_max = sec_max + total_sec_noise
print('noisy maximum:', mpc.run(mpc.output(noisy_sec_max)))

# Reveal (=recombine the shares) of this noisy maximum 
noisy_max = float(mpc.run(mpc.output(noisy_sec_max, receivers=range(0,M))))

# If it is lower than a threshold, the teachers are not confident enough
# Abort the aggregation mechanism #TODO: pass on to NEXT votes, next predictions
if noisy_max < THRESHOLD:
    print(f'\n===== SMALLER THAN THRESHOLD ({THRESHOLD}) =====\n')
    print('--> ABORT\n')
    print('='*50)
    mpc.run(mpc.shutdown())
    print('='*50)

# If it is greater than the threshold (meaning that the teachers are confident),
# we should aggregate a noisy version of the votes of each party 
else:
    print(f'\n===== GREATER THAN THRESHOLD ({THRESHOLD}) =====\n')
    
    # Generate the vector of gaussian noise values
    noise_vector = np.random.normal(0, SIGMA2, NB_CLASSES)
    
    # Add it to the votes to noise them
    noisy_votes = list(map(float, np.array(my_votes) + noise_vector)) # Cast from elem type <numpy.float64> to normal python <float>

    # Convert them to secure type
    sec_noisy_votes = list(map(secfxp, noisy_votes))

    # Secret-share them with every party
    all_sec_noisy_votes = mpc.input(sec_noisy_votes)

    # Aggregate all these noisy votes from every party
    total_sec_noisy_votes = vector_add_all(all_sec_noisy_votes)
    print('total noisy votes:', mpc.run(mpc.output(total_sec_noisy_votes)))

    # Compute the secure argmax on this aggregated vector
    sec_argmax = argmax(total_sec_noisy_votes)
    
    # Our label is the revealed (=recombined) argmax
    label = mpc.run(mpc.output(sec_argmax))

    print('LABEL =', label)
    print('')
    
    print('='*50)
    mpc.run(mpc.shutdown())
    print('='*50)
