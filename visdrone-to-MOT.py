import numpy as np
import pandas as pd
import os
from PIL import Image

def old():
    splits = ['test-dev', 'train', 'val']

    for split in splits:
        print('Solving split:', split)
        data_dir = 'D:/thesis-data/VisDrone2019-MOT-' + split + '/annotations/'
        annotations = os.listdir('D:/thesis-data/VisDrone2019-MOT-' + split + '/annotations')

        for ann in annotations:
            data = np.loadtxt(data_dir + ann, delimiter=",", dtype=float)
            data[:,-3:] = -1
            np.savetxt('D:/thesis-data/VisDrone2019-MOT-' + split + '/annotations/' + ann, data.astype(float), fmt ='%.0f', delimiter=",")

def convert(bbox, img_size):
    #bbox top_left_x top_left_y width height
    dw = 1/(img_size[0])
    dh = 1/(img_size[1])
    x = bbox[0] + bbox[2]/2
    y = bbox[1] + bbox[3]/2
    x = x * dw
    y = y * dh
    w = bbox[2] * dw
    h = bbox[3] * dh

    return x,y,w,h

if __name__ == '__main__':
    path_im = 'D:/thesis-data/VisDrone2019-MOT-test-dev/sequences/'
    path_anno = 'D:/thesis-data/VisDrone2019-MOT-test-dev/annotations/'

    imgs = os.listdir(path_im)
    seqs = os.listdir(path_anno)

    for i in range(len(seqs)):
        current_annotation = pd.read_csv(path_anno + seqs[i], delimiter=',', header=None)
        current_annotation.columns = ['frame', 'object', 'min-x', 'min-y', 'width', 'height', 'score', 'object-category', 'truncation', 'occlusion']
        unique_frames = current_annotation['frame'].unique()

        current_images = os.listdir(path_im + '/' + imgs[i])
        bbox_list = list()

        for frame in unique_frames:
            current_frame = current_annotation[current_annotation['frame'] == frame]
            # Börja indexa från 0
            current_image = current_images[frame - 1]

            img = Image.open(path_im + imgs[i] + '/' + current_image)
            img_size = img.size

            for row in current_frame.values:
                x,y,w,h = convert(row[2:6], img_size)
                bbox_list.append([row[0],row[1],x,y,w,h, 1, -1, -1, -1])
        
        np.savetxt('D:/thesis-data/VisDrone2019-MOT-test-dev/annotations-yolobbox/'  + seqs[i][:-4] + '.txt', pd.DataFrame(bbox_list, columns = ['frame', 'object', 'x', 'y', 'w', 'h', 'conf', 'w1', 'w2', 'w3']).values, fmt='%1.6f',  delimiter=",")
