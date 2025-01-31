import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix

# Simple example matrix
vectors = np.array([
    [0, 0, 2, 3, 0],
    [4, 0, 0, 0, 6],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 7, 0, 8, 0]
])

# CSR format
sparse_csr = csr_matrix(vectors)
print("Compressed Sparse Row (CSR) Matrix:\n", sparse_csr)

'''
Compressed Sparse Row (CSR) Matrix:
   (0, 2)	2
  (0, 3)	3
  (1, 0)	4
  (1, 4)	6
  (4, 1)	7
  (4, 3)	8
'''

# CSC format
sparse_csc = csc_matrix(vectors)
print("Compressed Sparse Column (CSC) Matrix:\n", sparse_csc)

'''
Compressed Sparse Column (CSC) Matrix:
   (1, 0)	4
  (4, 1)	7
  (0, 2)	2
  (0, 3)	3
  (4, 3)	8
  (1, 4)	6
'''

# COO format
sparse_coo = coo_matrix(vectors)
print("Coordinate Format (COO) Matrix:\n", sparse_coo)

'''
Coordinate Format (COO) Matrix:
   (0, 2)	2
  (0, 3)	3
  (1, 0)	4
  (1, 4)	6
  (4, 1)	7
  (4, 3)	8
'''

# Running operations on CSR and CSC matrices
weighted_csr = sparse_csr.multiply(0.5)
print("Weighted CSR:\n", weighted_csr.toarray())

'''
Weighted CSR:
 [[0.  0.  1.  1.5 0. ]
 [2.  0.  0.  0.  3. ]
 [0.  3.5 0.  4.  0. ]
 [0.  0.  0.  0.  0. ]
 [0.  0.  0.  0.  0. ]]
'''

# Operation on COO requires conversion to CSR or CSC first
weighted_coo = sparse_coo.tocsr().multiply(0.5)
print("Weighted COO:\n", weighted_coo.toarray())

'''
Weighted COO:
 [[0.  0.  1.  1.5 0. ]
 [2.  0.  0.  0.  3. ]
 [0.  3.5 0.  4.  0. ]
 [0.  0.  0.  0.  0. ]
 [0.  0.  0.  0.  0. ]]
'''