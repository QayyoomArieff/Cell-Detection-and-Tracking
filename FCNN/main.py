import tensorflow as tf
import numpy as np
import scipy
import keras
import os
from scipy import misc
from scipy import signal
import scipy.ndimage as ndimage
from PIL import Image
import matplotlib.cm as cm
import matplotlib.axes
from matplotlib import pyplot as plt
from keras import backend as K
import cv2
import CNN
from skimage.feature import peak_local_max
import sklearn.mixture
import Lineage
import LineageTree
import Bio
from Bio import Phylo
import io
import os
import scipy.stats

# Reads the input video
def Load(path):
    timelapse = []

    try:
        vid = cv2.VideoCapture(path)
        hasnext, frame = vid.read()


        while hasnext:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            timelapse.append(frame)
            hasnext, frame = vid.read()

    except:
        timelapse.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    return timelapse


def Centroid(density, original, count, previous, id, tag):
    # Calculates centroids

    coords = peak_local_max(density, threshold_abs=0.2, min_distance=8)#, num_peaks=count)
    original = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    coords = np.ndarray.tolist(coords)
    coords, id = Lineage.Link(coords, previous, id, tag, maxdist=200)


    for i in coords:
        if len(i)>2 and i[2][0]=='6':
            original = cv2.drawMarker(original, position=(i[1], i[0]), color=(0, 140, 255), markerType=cv2.MARKER_CROSS,
                                      markerSize=8, thickness=2)

        else:
            original = cv2.drawMarker(original, position=(i[1], i[0]), color=(255, 80, 20), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=2)

        # Uncomment to show cell ID's in the video
        #if len(i)>2:
            #original = cv2.putText(original, text=str(i[3]), org=(i[1]-4, i[0]+15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 0, 255))

    original = cv2.putText(original, text= "Cell Count = "+str(count), org=(10,40) ,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 140, 255))


    return coords, original, id


def main(path, weights):
    timelapse = Load(path)
    previous = None
    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)

    DensityNet = CNN.Density()
    DensityNet.load_weights("weights/"+weights+".h5")

    vid = cv2.VideoWriter(path[:-4] + "_Cross.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, ((timelapse[0].shape[1]//4)*4, (timelapse[0].shape[0]//4)*4))
    id = 0
    trees = {}
    frameno=0
    tag=1

    if timelapse:
        mean = np.mean(timelapse)
        std = np.std(timelapse)

    for image in timelapse:

        frame = (image-mean)/std
        frame = np.expand_dims(frame, 0)
        frame = np.expand_dims(frame, -1)

        density = DensityNet.predict(frame)
        density = density.squeeze(axis=(0, -1))
        count = int(np.sum(density)/100)    # Calculate cell count

        # Scale output density map
        factor = 255.0/np.amax(density)
        bwdensity = (density * factor)
        colour = cv2.applyColorMap(bwdensity.astype(np.uint8), cv2.COLORMAP_JET)

        dot, cross, id = Centroid(density, image, count, previous, id, tag)

        vid.write(cross)

        if dot:
            previous = dot


        cv2.imshow("1", colour); cv2.imshow("2", cross); cv2.waitKey(0)


        # Add centroids to lineage tree

        for cell in dot:
            tree = cell[2].find(".")

            if tree == -1:
                tree = cell[2]
            else:
                tree = cell[2][:tree]

            if tree not in trees:
                trees[tree] = LineageTree.LineageTree(cell, frameno)
                tag+=1

            else:
                trees[tree].insert(cell)

        frameno+=1

    # Display lineage tree
    for i in trees:
        handle = io.StringIO(str(trees[i]))
        tree = Phylo.read(handle, "newick")

        ax = Phylo.draw(tree, do_show=False)
        ax.ticklabel_format(axis='x', useOffset=-trees[i].frameno)
        ax.set_xlabel('Frame Number')
        plt.show()

    return trees


# Call Main with the path to the video and the type of cell out of the pre-trained weights
if __name__ == "__main__":
    # main("lineagecrop1.avi", "Msmegmatis")



