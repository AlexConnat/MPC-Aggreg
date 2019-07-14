#!/usr/bin/python3

from mpyc.runtime import mpc

# Iteratively call mpc.vector_add() to add all lists in the given list
# vectors = [ [1,2,3,4,5], [1,1,0,1,1], [10,20,30,40,50] ] --> returns [12,23,33,45,56]
def vector_add_all(vectors):

    if len(vectors) == 1:
        return vectors[0]

    v = mpc.vector_add(vectors[0], vectors[1])

    for i in range(2, len(vectors)):
        v = mpc.vector_add(v, vectors[i])

    return v


def scalar_add_all(scalars):

    if len(scalars) == 1:
        return scalars[0]

    S = mpc.add(scalars[0], scalars[1])

    for i in range(2, len(scalars)):
        S = mpc.add(S, scalars[i])

    return S


def argmax(x):
    argmax = type(x[0])(0)
    m = x[0]
    for i in range(1, len(x)):
        b = (m >= x[i])
        argmax = mpc.if_else(b, argmax, i)
        m = mpc.if_else(b, m, x[i])
    return argmax
