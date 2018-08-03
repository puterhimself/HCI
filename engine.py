import os
import cv2
import time
import keras
import numpy as np
from PIL import Image
from keras import models
from pymouse import PyMouse
from pandas import read_csv
from pykeyboard import PyKeyboard
from sklearn.utils import shuffle
from keras.models import load_model
from matplotlib import pyplot as plt
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Activation, Dropout, Flatten, MaxPooling2D

# variables

saveIMG = False
X, Y = 900, 172
height, width = 320, 320
row, cols, channels = 320, 320, 1
quack, guess, prob = False, None, None
counter, sampleSize = 0, 80
batch, epoch, filters, pool, conv, = 32, 5, 32, 2, 3
mouse, keyboard, cap = PyMouse(), PyKeyboard(), cv2.VideoCapture(0)
GestureName, gesttureDir, scriptsDir, modelFile, summary = 'nothing', './Gestures/', './Scripts/', 'model.h5', ''

with open('GesturesList.lel', 'r') as file:
    g = file.read()
    gestureList = g.split()
    gestureList.sort()
    classes = len(gestureList)

banner = '''\n
    What would you like to do ?
    1- Use pretrained model for gesture recognition
    2- Train the model (you will require image samples for training under Gestures-)
    '''

# Function


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
    time.sleep(0.05)


def NewGesture():
    global GestureName, saveIMG, classes, gestureList, retErr

    # if GestureName == '':
    #     GestureName = input('Gesture Name:').strip().lower()
        # time.sleep(5)

    if GestureName in gestureList:
        GestureName = ''
        retErr = "Gesture already exists"

    else:
        saveIMG = True
        gestureList.append(GestureName)
        with open('GesturesList.lel', 'a') as f:
            f.write(GestureName + '\n')
        classes = len(gestureList)
        with open(scriptsDir+GestureName, 'w') as file:
            file.write(GestureName + "commands")


def init(path):
    global classes
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
    TrainData = [data, labels]
    # print(labels)
    (X, y) = (TrainData[0], TrainData[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=69)

    X_train = X_train.reshape(X_train.shape[0], row, cols, channels)
    X_test = X_test.reshape(X_test.shape[0], row, cols, channels)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(y_train, classes)
    Y_test = to_categorical(y_test, classes)

    return X_train, X_test, Y_train, Y_test


def loadNN():
    global summary
    model = models.Sequential()
    #padding:valid
    model.add(Conv2D(filters, (conv, conv), padding='same', input_shape=(row, cols, channels), activation='relu'))
    model.add(Conv2D(filters, (conv, conv), activation='relu'))
    model.add(MaxPooling2D(pool_size=(pool, pool)))
    model.add(Dropout(0.5))

    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.01, decay=0.01/epoch)
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    summary = model.summary()
    model.get_config()

    return model


def TrainModel(model):
    X_train, X_test, Y_train, Y_test = init(gesttureDir)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.4,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(X_train)
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch),
                                  verbose=2, epochs=epoch, validation_data=(X_test, Y_test), workers=4)

    # history = model.fit(X_train, Y_train, batch_size=batch, epochs=epoch, verbose=1, validation_split=0.2)
    # model.save_weights("Weight.hdf5", overwrite=True)
    # modelFile = model.to_json()
    # with open('model.json', 'w') as file:
    #     json.dump(modelFile, file)
    model.save(modelFile)

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
    plt.show()
    # Evaluate The Trained Model
    [test_loss, test_acc] = model.evaluate(X_test, Y_test)
    print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


def prediction(model, frame):
    # global guess
    if model is None:
        return
    image = np.array(frame).flatten()
    image = image.reshape(row, cols, channels)
    image = image.astype('float32')
    image /= 255
    NNimage = image.reshape(1, row, cols, channels)
    predicted = model.predict_classes(NNimage)
    predict = model.predict(NNimage)

    return int(predicted[0]), predict


def execute(index):
    global quack
    if quack:
        pass


def binaryMask(frame, x, y, w, h):
    global saveIMG
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(thresh, 70, 255,  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if saveIMG:
        GestureImage(res)
    return res


def main():
    global cap, X, Y, height, width, saveIMG, guess, prob

    # Input = int(input(banner))
    # if Input == 1:
    model = load_model(modelFile)

    # elif Input == 2:
    #     model = loadNN()
    #     TrainModel(model)

    # elif Input == 3:
    #     model = None

    ret = cap.set(3, 1280)
    ret = cap.set(4, 720)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, flipCode=3)
        if ret:
            roi = binaryMask(frame, X, Y, width, height)

        # guess, prob = prediction(model, roi)
        cv2.imshow("Output", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

        elif key == ord('s'):
            NewGesture()

        # return roi

        elif key == ord('g'):
        #     guess = True
            guess, _ = prediction(model, roi)
        #     execute(guess)
            print(guess)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # model = loadNN()
    # model = load_model(modelFile)
    # img = cv2.imread(gesttureDir + 'ok1.png', 0)
    # _, p = prediction(model, img)
    # print(p)
    # _, x, _, _ = init(gesttureDir)
    # print(x, model.predict_classes(x=x, batch_size=batch, verbose=2))
    # TrainModel(model)
    # NewGesture()
    main()