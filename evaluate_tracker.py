import subprocess
import os
import shutil
import numpy as np

import motmetrics as mm

if __name__ == '__main__':
    data_dir = 'D:/thesis-data/VisDrone2019-MOT-test-dev/sequences/'
    yolo_model = 'model-final.pt'
    deep_sort_model = 'resnet50'

    sequences = os.listdir('D:/thesis-data/VisDrone2019-MOT-test-dev/sequences')
    shutil.rmtree('evaluated_sequences')
    os.mkdir('evaluated_sequences')

    # loop across all test sequences
    for seq in sequences:
        childproc = subprocess.Popen('python track.py --source ' + data_dir + seq + \
                                     ' --yolo_model ' + yolo_model + \
                                     ' --deep_sort_model ' + deep_sort_model + \
                                     ' --project evaluated_sequences/ --name "sequence" --save-txt')
        childproc.wait()
        shutil.move("evaluated_sequences/sequence/" + str(seq) + ".txt",
                    "evaluated_sequences/")
        shutil.rmtree("evaluated_sequences/sequence")
        pred = np.loadtxt("evaluated_sequences/" + seq + ".txt", dtype=int)
        np.savetxt("evaluated_sequences/" + seq + ".txt", pred.astype(float), fmt ='%.0f', delimiter=",")

    #subprocess.Popen('python visdrone-to-MOT.py')
