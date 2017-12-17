import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

import tensorflow as tf

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools


def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W, padd):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


# local layer weight initialization
def local_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.04)
    return tf.Variable(initial)

def local_bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# serve data by batches
def next_batch(batch_size):
    
    global train_images
    global train_labels
    global index_in_epoch
    global epochs_completed
    
    start = index_in_epoch
    index_in_epoch += batch_size
    
    # when all trainig data have been already used, it is reorder randomly    
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig2 = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, int(cm[i, j]*100)/100.0,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig2.savefig("confuMatrix.png")




pandaRead=pd.read_csv('dataGray.csv', sep=',',header=None)
adata = pandaRead.as_matrix()

X = adata[:,0:64*64]
y = adata[:,64*64]

num_all = X.shape[0]
print 'number of all the data: '+str(num_all)

X = X.astype(np.float)

X = X - X.mean(axis=1).reshape(-1,1)
X = np.multiply(X,100.0/255.0)

each_pixel_mean = X.mean(axis=0)
each_pixel_std = np.std(X, axis=0)

images = np.divide(np.subtract(X,each_pixel_mean), each_pixel_std)

labels_count = 11
print 'The number of different facial expressions is %d'%labels_count

TEST_SIZE = 700
test_ind = np.random.choice(np.arange(num_all),TEST_SIZE,replace=False)

test_labels = y[test_ind]
test_images = images[test_ind,:]

print 'the shape of test data: '+str(test_images.shape)
print 'the shape of test label: '+str(test_labels.shape)

train_val_images = np.delete(images, test_ind, axis=0)
train_val_labels = np.delete(y, test_ind)

labels = dense_to_one_hot(train_val_labels, labels_count)
labels = labels.astype(np.uint8)

VALIDATION_SIZE = 300

train_val_size = train_val_images.shape[0]

val_ind = np.random.choice(np.arange(train_val_size),VALIDATION_SIZE,replace=False)

validation_labels = labels[val_ind,:]
validation_images = train_val_images[val_ind,:]

print 'the shape of validation data: '+str(validation_images.shape)
print 'the shape of validation label: '+str(validation_labels.shape)

train_labels = np.delete(labels, val_ind, axis=0)
train_images = np.delete(train_val_images, val_ind, axis=0)

print 'the shape of training data: '+str(train_images.shape)
print 'the shape of training label: '+str(train_labels.shape)

print 'The number of final training data: %d'%(len(train_images))

# input & output of NN

image_pixels = images.shape[1]
# images
x = tf.placeholder('float', shape=[None, image_pixels])
# labels
y_ = tf.placeholder('float', shape=[None, labels_count])

# first convolutional layer 64
W_conv1 = weight_variable([5, 5, 1, 64])
b_conv1 = bias_variable([64])

image_width = 64
image_height = 64

# (1900, 128*128*3) => (1900,128,128,3)
image = tf.reshape(x, [-1,image_width , image_height,1])
#print (image.get_shape()) # =>(27000,48,48,1)


h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
#print (h_conv1.get_shape()) # => (1900,64,64,64)
h_pool1 = max_pool_2x2(h_conv1)
#print (h_pool1.get_shape()) # => (1900,32,32,64)
h_norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

# second convolutional layer
W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])

h_conv2 = tf.nn.relu(conv2d(h_norm1, W_conv2, "SAME") + b_conv2)
#print (h_conv2.get_shape()) # => (1900,32,32,128)

h_norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

h_pool2 = max_pool_2x2(h_norm2)

# densely connected layer local 3
W_fc1 = local_weight_variable([16 * 16 * 128, 3072])
b_fc1 = local_bias_variable([3072])

# (27000, 12, 12, 128) => (27000, 12 * 12 * 128)
h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 128])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#print (h_fc1.get_shape()) # => (27000, 1024)

# densely connected layer local 4
W_fc2 = local_weight_variable([3072, 1536])
b_fc2 = local_bias_variable([1536])

# (40000, 7, 7, 64) => (40000, 3136)
h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])

h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)
#print (h_fc1.get_shape()) # => (40000, 1024)

# dropout
keep_prob = tf.placeholder('float')
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# readout layer for deep net
W_fc3 = weight_variable([1536, labels_count])
b_fc3 = bias_variable([labels_count])

y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

#print (y.get_shape()) # => (40000, 10)

# settings
LEARNING_RATE = 1e-4

# cost function
cross_entropy = -tf.reduce_sum(y_*tf.log(y))


# optimisation function
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

# evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

# prediction function
#[0.1, 0.9, 0.2, 0.1, 0.1 0.3, 0.5, 0.1, 0.2, 0.3] => 1
predict = tf.argmax(y,1)

# set to 3000 iterations 
TRAINING_ITERATIONS = 3000
    
DROPOUT = 0.5
BATCH_SIZE = 100

epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# start TensorFlow session
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []
x_range = []

display_step=100

for i in range(TRAINING_ITERATIONS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
        
        train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
                                                  y_: batch_ys, 
                                                  keep_prob: 1.0})       
        if(VALIDATION_SIZE):
            valida_ind = np.random.choice(np.arange(VALIDATION_SIZE),BATCH_SIZE,replace=False)
            validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[valida_ind], 
                                                            y_: validation_labels[valida_ind], 
                                                            keep_prob: 1.0})                                  
            print('training_accuracy / validation_accuracy => %.4f / %.4f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        x_range.append(i)
        
        # increase display_step
        if i%(display_step*10) == 0 and i and display_step<100:
            display_step *= 10
    # train on batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

print 'finish training'

# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={x: validation_images, 
                                                   y_: validation_labels, 
                                                   keep_prob: 1.0})
    fig1 = plt.figure()
    print('validation_accuracy => %.4f'%validation_accuracy)
    plt.plot(x_range, train_accuracies,'-b', label='Training')
    plt.plot(x_range, validation_accuracies,'-g', label='Validation')
    plt.legend(loc='lower right', frameon=False)
    plt.ylim(ymax = 1.0, ymin = 0.0)
    plt.ylabel('accuracy')
    plt.xlabel('step')
    fig1.savefig("test.png")
    #plt.show()

predicted_lables = np.zeros(test_images.shape[0])

print 'testing'
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], 
                                                                                keep_prob: 1.0})

print 'test accuracy:'
print ('predicted_lables({0})'.format(len(predicted_lables)))

print accuracy_score(test_labels, predicted_lables)

confusion_matrix(test_labels, predicted_lables)

cnf_matrix = confusion_matrix(test_labels, predicted_lables)

class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain', 'NonFace']

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion Matrix for Test Dataset')



