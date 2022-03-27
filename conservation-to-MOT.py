import os
import pandas as pd
import numpy as np
import shutil
from PIL import Image

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

def con_to_mot():
    path_im = 'D:/thesis-data/ConservationDrones/TestReal/images/'
    path_anno = 'D:/thesis-data/ConservationDrones/TestReal/annotations/'
    target_dir = 'D:/thesis-data/ConservationDrones-MOT/'

    imgs = os.listdir(path_im)
    seqs = os.listdir(path_anno)

    for i in range(len(seqs)):
        current_annotation = pd.read_csv(path_anno + seqs[i], delimiter=',', header=None)
        current_annotation.columns = ['frame', 'object', 'min-x', 'min-y', 'width', 'height', 'object-class', 'species', 'occluded', 'noisy']
        unique_frames = current_annotation['frame'].unique()

        current_images = os.listdir(path_im + '/' + imgs[i])
        bbox_list = list()

        for frame in unique_frames:
            current_frame = current_annotation[current_annotation['frame'] == frame]
            current_image = current_images[frame]

            for row in current_frame.values:
                bbox_list.append([row[0]+1,row[1],row[2],row[3],row[4],row[5], 1, -1, -1, -1])
            
            if not os.path.exists(target_dir + 'images' + '/' + seqs[i][:-4]):
                os.mkdir(target_dir + 'images' + '/' + seqs[i][:-4])

            shutil.copy(path_im + seqs[i][:-4] + '/' + current_image, target_dir + '/images/' + seqs[i][:-4] + '/' + current_image)

        np.savetxt(target_dir + 'annotations' + '/' +  seqs[i][:-4] + '.txt', pd.DataFrame(bbox_list, columns = ['frame', 'object', 'x', 'y', 'w', 'h', 'conf', 'w1', 'w2', 'w3']).values, fmt='%1.6f', delimiter=",")


if __name__ == '__main__':
    con_to_mot()


    '''
    anns = os.listdir('D:/thesis-data/ConservationDrones-MOT/annotations')
    for ann in anns:
        current_annotation = pd.read_csv('D:/thesis-data/ConservationDrones-MOT/annotations/' + ann, delimiter=' ', header=None)
        current_annotation.columns = ['frame', 'object', 'x', 'y', 'w', 'h', 'object-class', 'w1', 'w2', 'w3']
        unique_frames = current_annotation['frame'].unique()

        num_imgs = len(os.listdir('D:/thesis-data/ConservationDrones-MOT/images/' + ann[:-4]))

        print(len(unique_frames))
        print(num_imgs)

        print('\n')'''