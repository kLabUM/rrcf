import numpy as np

def InsertPoint_cut(S ,p):
    """
    Generates the cut dimension and cut value
    based on the Insertpoint algorithm
    ----
    Inputs:
    S : Set of point to be split (numpy array (n x d))
    p : New point to be inserted (numpy array (1 x d))

    Returs:
    dimenstion for cut, cut value
    ----
    Example:
    InsertPoint_cut(x_inital, x_new)
    (0, 0.9758881798109296)
    """
    # Generate the bounding box

    # Note that the box is first generated based on the inital data
    B_s = np.zeros((S.shape[1],2))# [min, max] of dx2
    B_s[:,0] = np.min(S, axis=0) # min of column x_inital
    B_s[:,1] = np.max(S, axis=0) # max of column x_inital

    # Update the bounding box based on the internal point

    # This is stupid ! why not just do it in the above.
    B_s[:,0] = np.minimum(B_s[:,0], p) # compare element wise
    B_s[:,1] = np.maximum(B_s[:,1], p)

    # Find r
    temp_sum = sum(B_s[:,1]-B_s[:,0])
    # resolution of r increases as the r magnitude increases, 10 times for now.
    temp_len = np.linspace(0, temp_sum, int(temp_sum*10))
    r = np.random.choice(temp_len, 1)[0] # returns a float.

    # Identify the cut.
    temp_diff = B_s[:,1]-B_s[:,0]
    obj = np.cumsum(temp_diff)

    cut_dimenstion = 9999 # Hope we do not have 999 dimentional data
    for i in range(0,len(obj)):
        if obj[i] >= r:
            cut_dimenstion = i
            break

    cut = B_s[j,0] + obj[j] - r

    return cut_dimenstion, cut
