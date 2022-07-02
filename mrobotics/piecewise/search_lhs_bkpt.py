""" Search for the curve segment/ piece given a test curve parameter `query_arclen`
More precisely,
* we assume the curve segments contains the `idx2arclen` table
  which is the sequence of "breakpoints" along the curve,
  i.e. it is assumed to be distinct^ and sorted in ascending order
        
  ^ locally distinct along the curve parameter domain

* we aim to search for the largest index i in `idx2arclen`
  such that idx2arclen[i] <= query_arclen (the equality shall rarely happen due to numeric reasons)

* return the following index in case that fall beyond the curve's domain:
  -1 if query_arclen < 0.0 
  -2 if query_arclen > tot_distance
"""

# what if arclen == query?
# what if arclen out of range/ exactly on the boundary?

def linear_search(idx2arclen, query_arclen):
    for i, arclen in enumerate(idx2arclen):
        if arclen > query_arclen: 
            return i-1 # in case of query_arclen < 0.0, require lower extrapolation
    return -2          # in case of query_arclen > (our implementation: >=) total_distance requires uppper extraoikatuib

def bisection(idx2arclen, query_arclen):
    pass

if __name__ == "__main__":
    assert linear_search([2.3,4.2, 10.4], 5.5) == 1
    assert linear_search([2.3,14.2, 110.4,342.2], 5.5) == 0
    assert linear_search([2.3,14.2, 110.4,342.2], 1.5) == -1
    assert linear_search([2.3,14.2, 110.4,342.2], 350.5) == -2

    import numpy as np
    assert linear_search(np.array([2.3,4.2, 10.4]), 5.5) == 1
