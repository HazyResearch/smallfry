import tests
import smallfry as sfry
import numpy as np


# Generate a random uniform matrix
unimat = 2*np.random.random(1000,50)-1

np.savetxt("uni.mat",unimat,fmt='%.12f')





