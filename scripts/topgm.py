#!/usr/bin/env python3

def build_pgm(map, comment='your ad here'):
  nrows, ncols = map.shape
  body = '\n'.join(' '.join(str(x) for x in map[i,:]) for i in range(nrows))
  return f'''P2
#{comment}
{ncols} {nrows}
65535
{body}'''

import glob, os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage {sys.argv[0]} basename')
        sys.exit(1)

    basename = sys.argv[1]

    files = glob.glob(f'{basename}????_?.tif')
    for fn in files:
        bn = os.path.splitext(fn)[0]
        print(f'Doing {bn}.pgm')
        cb = cv2.imread(fn, cv2.IMREAD_ANYDEPTH)
        if cb.shape[1] % 2 == 1:
            # do actual width padding instead
            cb = cb[:,:-1]
        if cb.shape[0] % 2 == 1:
            # do actual width padding instead
            cb = cb[:-1,:]
        open(f'{bn}.pgm','w').write(build_pgm(cb, bn))

