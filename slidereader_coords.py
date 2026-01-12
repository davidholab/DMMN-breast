from pdb import set_trace
import os
import numpy as np
import openslide
import extract_tissue
import shutil
import math
import glob

if __name__ == '__main__':

    slides_to_read = ["./testing_image/1.svs"]  # list of testing whole slide images    

    ### get coordinates of tissue patches in whole slide images using Otsu algorithm ###
    for image_path in slides_to_read:
        coord_path = image_path.replace("svs", "csv")
        coord_file = open(coord_path, 'w') # a file listing all patch coordinates
        slide = openslide.OpenSlide(image_path)
        slide_id = os.path.splitext(os.path.basename(image_path))[0]
        grid, _ = extract_tissue.make_sample_grid(slide, 512,
               40, 10, 10, False, prune=False, overlap=0)
        for (x,y) in grid:
            coord_file.write('{},{},{},0\n'.format(slide_id, x, y))