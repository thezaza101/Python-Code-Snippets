
import sys
import math
import numpy as np
from numba import jit
from PIL import Image
from itertools import repeat
from multiprocessing import Pool

@jit
def get_col(args):
    iy, width, height, max_iterations = args
    result = np.zeros((1, width))    
    for ix in np.arange(width):
        x0 = ix*3.0/width - 2.0
        y0 = iy*2.0/height - 1.0
        x = 0.0
        y = 0.0
        for iteration in range(max_iterations):
            x_new = x*x - y*y + x0
            y = 2*x*y + y0
            x = x_new

            if x*x + y*y > 4.0:
                # color using pretty linear gradient
                color = 1.0 - 0.02*(iteration - np.log2(np.log2(x*x + y*y)))
                break
        else:
            # failed, set color to black
            color = 0.0

        result[0, ix] = color
    return result

if __name__ == '__main__':
    
    height = int(sys.argv[1])
    width = int(height * 1.5)
    print(f'Attempting to image of size {width} x {height}. This will result in a {'{:,}'.format(width*height)} pixel image!')
    max_iterations = 255    
    result = np.zeros((height, width))
    processes = int(sys.argv[2])
    po = Pool(processes)    
    iy = np.arange(height)
    test = po.map_async(get_col, zip(iy, repeat(width), repeat(height), repeat(max_iterations) )).get()
    for i in np.arange(height):
        result[i,:] = test[i]

    mandelbrot = result
    mandelbrot = np.clip(mandelbrot*255, 0, 255).astype(np.uint8)
    mandelbrot = Image.fromarray(mandelbrot)
    
    mandelbrot.save(f'{sys.argv[3]}.png')
    print(f'saved {sys.argv[3]}.png')