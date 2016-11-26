import os.path
from app.find_mapping import Lexconn
import numpy as np

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

if __name__ == '__main__':
    a_dir = '/Users/majid/Documents/git/disco-parallel/parallel.corpus/outputs/leave-one-out/'
    for fld in get_immediate_subdirectories(a_dir):
        print(fld)
        emission = np.loadtxt(os.path.join(a_dir, fld, 'emission.txt'))
        lexconn = np.loadtxt(os.path.join(a_dir, fld,'entries.txt'))
        print(lexconn.shape)
        print(emission.shape)
        print(np.max(lexconn, axis=0))

        emission = emission.astype(np.float32)
        lexconnEntries = lexconn.astype(np.float32)
        lexconn = Lexconn(emission, lexconnEntries)
        got_mapping = lexconn.get_mapping(20000)
        print(got_mapping)
        np.savetxt(os.path.join(a_dir, fld, 'mapping.txt'), got_mapping)
        