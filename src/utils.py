"""
utils.py
"""

import numpy as np
import struct
import sys
import io

###############################################################################

"""
Load a QUANT format matrix into python.
A QUANT matrix stores the row count (m), column count (n) and then m x n IEEE754 floats (4 byte) of data
"""
def loadQUANTMatrix(filename):
    with open(filename,'rb') as f:
        (m,) = struct.unpack('i', f.read(4))
        (n,) = struct.unpack('i', f.read(4))
        print("loadQUANTMatrix::m=",m,"n=",n)
        matrix = np.arange(m*n,dtype=np.float64).reshape(m, n) #and hopefully m===n, but I'm not relying on it
        for i in range(0,m):
            data = struct.unpack('{0}f'.format(n), f.read(4*n)) #read a row back from the stream - data=list of floats
            for j in range(0,n):
                matrix[i,j] = data[j]
    return matrix

#def loadQUANTMatrixURL(url):
#    print("downloading from ",url)
#    response = http.request("GET", url)
#    with io.BytesIO(response.data) as f:
#        (m,) = struct.unpack('i', f.read(4))
#        (n,) = struct.unpack('i', f.read(4))
#        print("loadQUANTMatrix::m=",m,"n=",n)
#        matrix = np.arange(m*n,dtype=np.float64).reshape(m, n) #and hopefully m===n, but I'm not relying on it
#        for i in range(0,m):
#            data = struct.unpack('{0}f'.format(n), f.read(4*n)) #read a row back from the stream - data=list of floats
#            for j in range(0,n):
#                matrix[i,j] = data[j]
#    return matrix


###############################################################################
