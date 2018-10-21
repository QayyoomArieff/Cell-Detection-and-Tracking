import cv2
from skimage.feature import peak_local_max
import Lineage
import numpy as np
import CNN


# Calculate F1 score
def Fscore(density, gt, original,count, distance=5):
    TP = 0
    FP = 0
    FN = 0

    ignore = []

    densitycoords = peak_local_max(density, threshold_abs=0.4, min_distance=1)#, num_peaks=int(round(count)))
    densitycoords = np.ndarray.tolist(densitycoords)

    gtcoords = peak_local_max(gt)
    gtcoords = np.ndarray.tolist(gtcoords)

    gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    for i in gtcoords:
        gt = cv2.drawMarker(gt, position=(i[1], i[0]), color=(0, 140, 255), markerType=cv2.MARKER_CROSS,
                              markerSize=5, thickness=1)

    density = cv2.cvtColor(density, cv2.COLOR_GRAY2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    for i in densitycoords:
        density = cv2.drawMarker(density, position=(i[1], i[0]), color=(0, 140, 255), markerType=cv2.MARKER_CROSS,
                            markerSize=5, thickness=1)
        original = cv2.drawMarker(original, position=(i[1], i[0]), color=(0, 140, 255), markerType=cv2.MARKER_CROSS,
                                  markerSize=5, thickness=1)

    # Similar algorithm to lineage tracking
    for i in gtcoords:
        cell = Lineage.nearest(i, densitycoords, distance)

        if cell:
            TP += 1

            original = cv2.drawMarker(original, position=(cell[1], cell[0]), color=(255, 80, 20), markerType=cv2.MARKER_CROSS,
                                     markerSize=5, thickness=1)

            densitycoords.remove(cell)

        else:
            FN += 1

            original = cv2.drawMarker(original, position=(i[1], i[0]), color=(255, 255, 255), markerType=cv2.MARKER_CROSS,
                                    markerSize=5, thickness=1)

    FP = len(densitycoords)


    # TP FP FN
    print(TP, FP, FN, sep='\t')

    cv2.imshow("bwgt", original);cv2.waitKey(0)

    return TP, FP, FN


# Change to calculate F1 score of detection accuracy
path= "salmonella"

imlist = []
gtlist= []

for i in range (1,100):
    imlist.append((cv2.imread(path+str(i).zfill(3)+"cell.tif", cv2.IMREAD_GRAYSCALE)))
    gtlist.append((((cv2.imread(path+str(i).zfill(3)+"dots.png", cv2.IMREAD_GRAYSCALE))[:,:]>0)*100).astype(np.uint8))

DensityNet = CNN.Density()
DensityNet.load_weights("weights/salmonella.h5")    # Change Weights


mean = np.mean(imlist)
std = np.std(imlist)

for i in range(len(imlist)):
    imgorigin = imlist[i]
    img = imlist[i]
    gt = gtlist[i]


    img = (img-mean)/std
    img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    density = DensityNet.predict(img)

    density = density.squeeze(axis=(0, -1))
    count = np.sum(density)/100
    print(np.sum(gt)//100, count, sep='\t', end='\t')
    factor = 255.0/np.amax(density)
    bwdensity = (density * factor)
    colour = cv2.applyColorMap(bwdensity.astype(np.uint8), cv2.COLORMAP_JET)

    cv2.imshow("colour", colour); cv2.imshow("bw", bwdensity.astype(np.uint8))

    Fscore(density, gt, imgorigin,count,8)




