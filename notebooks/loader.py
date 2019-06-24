import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import time
from sklearn.model_selection import train_test_split
import h5py

#change folder path

DATADIR = "/home/leonardo/Downloads/BaseCancer"
CATEGORIES = ["benign", "malignant"]
training_data = []
label_data=[]
#IMG_SIZE = 256


def create_training_data():
    count = 0
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                #print (img_array)
                #new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append(img_array)
                label_data.append(class_num)
                count +=1
                if count == 100:
                    print("100 imágenes")
                    count=0
            except Exception as e:
                pass

    xTrain, xTest, yTrain, yTest = train_test_split(training_data, label_data, test_size = 0.25, random_state = 0)
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)
    yTest = np.array(yTest)

    with h5py.File('label_cancer.hdf5', 'w') as f:
        dset = f.create_dataset("default", data=label_data)

    return (xTrain, yTrain), (xTest, yTest)


def main():
    st = time.time()
    (x,y),(z,w) = create_training_data()
    elapsed_time = time.time() - st
    elapsed_time_minutes = int(int(elapsed_time) / 60)
    elapsed_time_seconds = int(elapsed_time) % 60
    print('loaded in %s [min] %s [s]' % (elapsed_time_minutes, elapsed_time_seconds))
    print(y.shape, x.shape)
    print(x[0,:,:,0])
    print (y[0], y[1], y[2], w[0], w[2])


if __name__  == '__main__':
    main()
