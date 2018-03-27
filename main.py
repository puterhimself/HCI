import cv2
import os
import time
import json
import numpy as np
from PIL import Image
from keras import models
from pymouse import PyMouse
from pykeyboard import PyKeyboard
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,Dense,Activation,Dropout,Flatten,MaxPooling2D

# variables

X, Y = 320, 172
row, cols, channels = 960, 540, 1
height, width = 540, 960

mouse, keyboard, cap = PyMouse(), PyKeyboard(), cv2.VideoCapture(0)
counter, sampleSize = 0, 301
saveIMG = False
batch, epoch, filters, pool, conv, classes = 32, 10, 32, 2, 3, 1
GestureName, gesttureDir, scriptsDir = '', './Gestures/', './Scripts/'

banner = '''\n
    What would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    '''

# Function


def init(path):
    ImageList = listDir(path)
    TotalImages = len(ImageList)
    ImageMatrix = np.array([
        np.array(Image.open(path + '/' + image).convert('L')).flatten()
        for image in ImageList], dtype='f')
    labels = np.ones(TotalImages, dtype=int)
    samplesPerClass = TotalImages // classes

    s, r = 0, samplesPerClass
    for i in range(classes):
        labels[s:r] = i
        s = r
        r = s + samplesPerClass

    data, labels = shuffle(ImageMatrix, labels, random_state=42)
    TrainData = [data,labels]
    (X, y) = (TrainData[0], TrainData[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
    X_train = X_train.reshape(X_train.shape[0], row, cols, channels)
    X_test = X_test.reshape(X_test.shape[0], row, cols, channels)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, classes)
    Y_test = np_utils.to_categorical(y_test, classes)

    return X_train, X_test, Y_train, Y_test


def listDir(path):
    listing = os.listdir(path)
    retList = list()
    for name in listing:
        if name.startswith('.'):
            continue
        retList.append(name)
    return retList


def GestureImage(img):
    global counter, GestureName, saveIMG

    if counter > (sampleSize - 1):
        counter = 0
        saveIMG = False
        GestureName = ''
        return
    counter = counter + 1
    Name = gesttureDir + GestureName + str(counter) + '.png'
    print("Saving img:", Name)
    cv2.imwrite(Name, img)
    time.sleep(0.04)


def NewGesture():
    global GestureName, saveIMG, classes

    if GestureName == '':
        GestureName = input('Gesture Name:')

    with open('GesturesList.json','r+') as file:
        gestureList = json.load(file)
        if GestureName in gestureList:
            print("Gesture already exsists")
            GestureName = ''
        else:
            saveIMG = True
            gestureList.append(GestureName)
            json.dump(gestureList,file)
            classes = len(gestureList)
            with open(scriptsDir+GestureName, 'w') as file:
                file.write(GestureName + "commands")


def loadNN():
    global getOutput
    model = models.Sequential()
    model.add(Conv2D(filters, (conv, conv), padding='valid', input_shape=(row, cols, channels),activation='relu'))
    model.add(Conv2D(filters, (conv, conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.summary()
    model.get_config()
    # model.load_weights(file)
    # layer = model.layers[11]
    # getOutput = K.function([model.layers[0].input,K.learning_phase()], [layer.output])

    return model


def TrainModel(model):
    X_train, X_test, Y_train, Y_test = init()
    history = model.fit(X_train, Y_train, batch_size=batch, epochs=epoch,verbose=1,validation_data=(X_test,Y_test))
    model.save_weights("newWeight.hdf5", overwrite=True)

    # Plot The Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Plot the Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    # Evaluate The Trained Model
    [test_loss, test_acc] = model.evaluate(X_test, Y_test)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


def prediction(model,frame):
    global getOutput
    image = np.array(frame).flatten()
    image = image.reshape(row, cols, channels)
    image = image.astype('float32')
    image /= 255
    NNimage = image.reshape(1, row, cols, channels)
    predicted = model.predict_classes(NNimage)

    return predicted


def binaryMask(frame, x, y, w, h):
    global saveIMG
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(thresh, 70, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if saveIMG:
        GestureImage(res)
    return res


def main():
    global cap, X, Y, height, width, saveIMG

    ret = cap.set(3, 1280)
    ret = cap.set(4, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame,flipCode=3)
        if ret:
            roi = binaryMask(frame, X, Y, width, height)
        cv2.imshow("original", frame)
        cv2.imshow("ROI",roi)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            NewGesture()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()