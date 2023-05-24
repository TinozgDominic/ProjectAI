import cv2
import numpy as np
import random

import matplotlib.pyplot as plt
import keras
from glob import glob

from keras import backend as K

from keras.applications import mobilenet
from keras.layers import Dense, dot,GlobalAveragePooling2D,Input, Lambda
from keras.models import Model
from keras.optimizers import RMSprop


'''Parameters'''

# data import options
class_list_filename = "myproducts_classes.txt"

# specify root directory where training / validation folders are located
root_dir = "data/myproduct/"
n_classes = 31

# training hyperparameters
n_epochs = 100
batch_size = 32
K.set_epsilon(1e-07)
epsilon = K.epsilon()

# save parameters
base_name = "MobileNet" # base network name corresponding to base_ids 0-3
date_dir = '22_05_2023' # specify save directory for results using YYYY-MM-DD format
similarity_type = 'cosine' # l1_sum, l2_sum, cosine
enable_model_save = True

'''Functions'''

# create a list of training image directory locations,class labels and image file formats
def create_class_list(list_filename):

    with open(list_filename, 'r') as f:
        list = []

        counter = 0
        for line in f.readlines():
            category, newLine = line.strip().split(',')
            list.append(category)
            counter+=1
    
    return list

# create numpy array of images from a specified class directory
def create_trainval(class_list, dataset_type):

    img_array = []
    labels_array = []

    counter = 0

    for i, class_name in enumerate(class_list):
        folder_name = root_dir + dataset_type + "/" + class_name

        for fn in glob(folder_name+'/*.'+'jpg'):
            img_cv = cv2.imread(fn)
            img_cv = cv2.resize(img_cv,(224,224),img_cv)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            img_array.append(img_cv)
            labels_array.append(i)
            counter+=1

    return img_array, labels_array

# create positive / negative training example pairs
def create_pairs(x, idcs):

    pairs = []
    labels = [] # 1 = positive pair, 0 = negative pair

    # find the minimum number of examples per class
    n_min = (min([len(idcs[d]) for d in range(n_classes)]) - 1)

    for n in range(n_classes):

        print("\nClass %i" %n)

        counter_pos = 0
        counter_neg = 0

        # create n_min positive pairs and n_min negative pairs per class
        for i in range(n_min):

            # positive example
            z1, z2 = idcs[n][i], idcs[n][i+1]
            pairs += [[x[z1], x[z2]]]
            counter_pos += 1

            # negative example
            inc = random.randrange(1,n_classes)
            n_rand = (n+inc) % n_classes
            z1, z2 = idcs[n][i], idcs[n_rand][i]
            pairs += [[x[z1], x[z2]]]
            counter_neg += 1

            labels += [1,0]

        print("\t# of positive pairs: %i" % counter_pos)
        print("\t# of negative pairs: %i" % counter_neg)

    return np.array(pairs,dtype=np.float32), np.array(labels)

# create a base model that generates a (n,1) feature vector for an input image
def create_base_model(input_shape):
    # MobileNet
    base = mobilenet.MobileNet(input_shape = input_shape, weights = 'imagenet', include_top = False)
    base_name = 'MobileNet'

    print("\nBase Network: %s" %base_name)

    top = GlobalAveragePooling2D()(base.output)

    # freeze all layers in the base network
    for layers in base.layers:
        layers.trainable = False

    model = Model(inputs = base.input, outputs = top, name = 'base_model')

    return model

# calculate l1_norm b/t feature vector outputs from base network
def l1_distance(feat_vects):

    x1, x2 = feat_vects

    result = K.maximum(x=K.sum(x=K.abs(x1-x2), axis=1, keepdims=True), y=epsilon)

    return result

# calculate l2_distance b/t feature vector outputs from base network
def l2_distance(feat_vects):

    x1, x2 = feat_vects

    result = K.sqrt(K.maximum(x=K.sum(x=K.square(x1 - x2), axis=1, keepdims=True), y=epsilon))

    return result

# calculate cosine distance b/t feature vector outputs from base network
def cos_distance(feat_vects):

    x1, x2 = feat_vects

    result = K.maximum(x=dot(inputs=[x1, x2], axes=1, normalize=True), y=epsilon)

    return result

# create a siamese model that calculates similarity b/t two feature vectors
def create_siamese_model(encoding_shape):

    encoding_a = Input(shape = encoding_shape)
    encoding_b = Input(shape = encoding_shape)

    fc1_a = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_a')(encoding_a)
    fc1_b = Dense(units=2048, activation='relu', kernel_regularizer=None, name='fc1_b')(encoding_b)

    # distance = Lambda(function = l1_distance, name = 'l1_distance', )([fc1_a, fc1_b])
    # distance = Lambda(function = l2_distance, name = 'l2_distance', )([fc1_a, fc1_b])
    distance = Lambda(function = cos_distance, name = 'cos_distance', )([fc1_a, fc1_b])

    prediction = Dense(units=1, activation='sigmoid', kernel_regularizer=None, name='sigmoid')(distance)

    model = Model(inputs=[encoding_a, encoding_b], outputs=prediction, name='siamese_model')

    return model

# calculates accuracy between predicted and ground-truth similarities
def compute_accuracy(y_true, y_pred):

    prediction = y_pred.ravel() > 0.5

    return np.mean(prediction == y_true)

# Callback
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from IPython.display import clear_output

class DisplayCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)

callbacks = [DisplayCallback(), 
            EarlyStopping(patience = 20, verbose = 1),
            ReduceLROnPlateau(patience = 5, verbose = 1),
            ModelCheckpoint('MySiamese.h5', verbose = 1, save_best_only = True)]

'''Main'''

if __name__==("__main__"):

    K.clear_session()

    print("\nComparison metric: %s" %similarity_type)

    class_list = create_class_list(class_list_filename)

    # import original train and validation sets
    x_train_orig,  y_train_orig = create_trainval(class_list,"train")
    x_val_orig, y_val_orig = create_trainval(class_list,"val")

    # convert to numpy arrays
    x_train = np.array(x_train_orig,dtype=np.float32)
    x_val = np.array(x_val_orig, dtype=np.float32)
    y_train = np.array(y_train_orig, dtype=np.float32)
    y_train = np.reshape(y_train,newshape=(y_train.shape[0],1))
    y_val = np.array(y_val_orig, dtype=np.float32)
    y_val = np.reshape(y_val, newshape=(y_val.shape[0], 1))

    # Normalize
    x_train /= 255
    x_val /= 255
    print("Normalizing training & validation b/t [0,1]\n")

    # create base_network
    base_model = create_base_model(input_shape=(224,224,3))

    # create encodings for train & validation data
    x_train_encoding = base_model.predict(x = x_train, batch_size = 1, verbose = 1)
    x_val_encoding = base_model.predict(x = x_val, batch_size = 1, verbose = 1)
    input_shape = x_train_encoding.shape[1:]

    print("\nEncoding shape: " + str(input_shape))
    print("Training encoding: " + str(x_train_encoding.shape))
    print("Validation encoding: " + str(x_val_encoding.shape))

    # find indices in loaded images in x_train & y_train corresponding to each class
    idcs_train = [np.where(y_train == i)[0] for i in range(n_classes)]
    idcs_val = [np.where(y_val == i)[0] for i in range(n_classes)]

    # create train & test pairs from original sets
    x_train_pair, y_train_pair = create_pairs(x_train_encoding,idcs_train)
    x_val_pair, y_val_pair = create_pairs(x_val_encoding, idcs_val)


    print("\nX-train pair size: "+str(x_train_pair.shape))
    print("Y-train pair Size: " + str(y_train_pair.shape))

    print("X-val pair size: " + str(x_val_pair.shape))
    print("Y-val pair size: " + str(y_val_pair.shape))

    print(x_train_pair[2].shape)
    print(y_train_pair[2])

    # create siamese model
    siamese_model = create_siamese_model(input_shape)
    print(siamese_model.summary())

    # binary cross entropy loss function
    siamese_model.compile(loss="binary_crossentropy", optimizer=RMSprop(), metrics=['accuracy'])
    history = siamese_model.fit(x = [x_train_pair[:,0], x_train_pair[:,1]],
                                y = y_train_pair,
                                batch_size = batch_size,
                                epochs = n_epochs,
                                validation_data = ([x_val_pair[:,0], x_val_pair[:,1]], y_val_pair),
                                shuffle = True,
                                verbose = True,
                                callbacks=callbacks)


    # compute accuracy on train &  validation sets
    print("\nPredictions with Trained Model:")
    y_pred_train = siamese_model.predict(x=[x_train_pair[:,0],x_train_pair[:,1]], verbose=True)
    y_pred_val = siamese_model.predict(x=[x_val_pair[:, 0], x_val_pair[:, 1]], verbose=True)

    acc_train = compute_accuracy(y_train_pair,y_pred_train)
    acc_val = compute_accuracy(y_val_pair,y_pred_val)
    print("Training set accuracy: %0.2f%%" % acc_train)
    print("Validation set accuracy: %0.2f%%" % acc_val)

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["loss"])
    plt.title("Training accuracy")

    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.legend(["Accuracy", "Loss"], loc = "lower right")

    plt.figure()

    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["val_loss"])
    plt.title("Validation accuracy")

    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    plt.legend(["Accuracy", "Loss"], loc = "lower right")
    plt.show()
    