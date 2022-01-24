import os
import pickle
from collections import namedtuple
import numpy as np
import constants as C

Point = namedtuple('Point', 'x, y')

MAX_DIST = np.sqrt((C.WORLD_SIZE[0]-0)**2+(C.WORLD_SIZE[1]-0)**2)
MAX_DIST_X = np.sqrt(C.WORLD_SIZE[0]**2)
MAX_DIST_Y = np.sqrt(C.WORLD_SIZE[1]**2)

def distance_between(pos1, pos2, normalize=False, justpos=None):
    # Calculates the distance between two positions
    dist = None
    max_dist = None
    if justpos==0:
        dist = pos1.x-pos2.x
        max_dist = MAX_DIST_X
    elif justpos==1:
        dist = pos1.y-pos2.y
        max_dist = MAX_DIST_Y
    else:
        dist = np.sqrt((pos1.x-pos2.x)**2+(pos1.y-pos2.y)**2)
        max_dist = MAX_DIST
    if normalize:
        dist /= max_dist
    return dist

def save(folder_path, file_name, data):
    # Pickle saves data to a file on disk
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = os.path.join(folder_path, file_name)
    file = open(file_name, 'wb')
    pickle.dump(data, file)
    file.close()

def load(folder_path, file_name):
    # Pickle loads something from a file on disk
    data = None
    if(os.path.exists(folder_path)):
        file_name = os.path.join(folder_path, file_name)
        if os.path.exists(file_name):
            file = open(file_name, 'rb')
            data = pickle.load(file)
            file.close()
        else:
            print("File does not exists: "+file_name)
    return data

def print_state(state, action,vals):
    # Prints agent's state in a readable way.
    for i, val in enumerate(state):
        print(C.STATE_LABELS[i]+":"+str(np.round(val,3)),end=', ')
    print('A='+C.ACTION_LABELS[action]+' ('+str(vals)+')')
    print('')

'''
def get_closest_dists_from_array(pos, pos_array, normalize = False):
    # Turns an array of positions into a 
    dists = [-C.WORLD_SIZE[0],C.WORLD_SIZE[0],-C.WORLD_SIZE[1],C.WORLD_SIZE[1]]
    for pt in pos_array:
        x_dist, y_dist = pos.x - pt.x, pos.y - pt.y
        if dists[0] < x_dist <= 0: #L
            dists[0] = x_dist
        if 0 <= x_dist < dists[1]: #R
            dists[1] = x_dist
        if dists[2] < y_dist <= 0: #U
            dists[2] = y_dist
        if 0 <= y_dist < dists[3]: #D
            dists[3] = y_dist
    if normalize:
        dists[0] /= C.WORLD_SIZE[0]
        dists[1] /= C.WORLD_SIZE[0]
        dists[2] /= C.WORLD_SIZE[1]
        dists[3] /= C.WORLD_SIZE[1]
    return np.abs(dists)
'''

