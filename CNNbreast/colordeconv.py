"""
H&E color deconvolution code
2017-06-16 mslee
"""

import os
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt

from cntk import input as cntk_input
from cntk import Trainer, cross_entropy_with_softmax, classification_error, softmax, relu
from cntk.debugging import set_computation_network_trace_level
from cntk.logging import ProgressPrinter, log_number_of_parameters
from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType
from cntk.learners import momentum_as_time_constant_schedule
from cntk.layers import default_options, Convolution, MaxPooling
from cntk.layers import BatchNormalization, Dense, Sequential, For
from cntk.initializer import glorot_uniform

from CNNFunctions import changeCvediaToCNTKmap, MixCNTKmap

def intensity_to_od(rgb_img):
    """Convert intensity image to OD image"""
    intensity = np.sum(rgb_img, 2)
    max_idx = np.unravel_index(intensity.argmax(), intensity.shape)
    return -np.log10(rgb_img / rgb_img[max_idx])


def deconv(od_img):
    """H&E colordeconvolution funciton"""
    absorption = np.array([[0.65, 0.7, 0.28], [0.07, 0.99, 0.11]])
    inv_absorption = np.linalg.pinv(absorption)[:, np.newaxis]
    he_img = np.einsum('abc,cde->abe', od_img, inv_absorption)
    return he_img

def rgb_to_he(rgb_img):
    return intensity_to_od(deconv(rgb_img))

def create_txt(od_img, label, filename):
    """Create cntk supporting txt file"""
    print("Saving", filename)
    labels = list(map(' '.join, np.eye(2, dtype=np.uint).astype(str)))
    label_str = labels[label]
    with open(filename, 'a+b') as f_header:
        np.savetxt(f_header, od_img[:, :, 0], fmt='%.15f',
                   newline=' ', comments='', header='|labels')
        np.savetxt(f_header, od_img[:, :, 1], fmt='%.15f',
                   newline=' ', comments='', footer='|features {}\n'.format(label_str))


class ImageMinibatch(object):
    """Data object for cntk training"""
    def __init__(self, width, height, channels, classes, num_samples, filename):
        self.width = width
        self.height = height
        self.channels = channels
        self.classes = classes
        self.filename = filename
        self.num_samples = num_samples
        self.cur_filepos = 0
        self.cur_fileline = 0

    def next_minibatch(self, batch_size, transform):
        """Get next minibacth"""
        #Initialize necesseary data.
        batch_size = min(batch_size, self.num_samples-self.cur_fileline)
        features = np.empty((0, self.channels, self.height, self.width), dtype=np.float32)
        #features = np.empty((0, self.width*self.height*self.channels), dtype=np.float32)
        labels = np.empty((0, self.classes), dtype=np.float32)
        one_hot = np.eye(2)

        # Read map file starting from the previous line and create a mini-bath data.
        with open(self.filename) as mapfile:
            mapfile.seek(self.cur_filepos)
            for _ in range(batch_size):
                imgfile, label = mapfile.readline().split('\t')
                #img = np.transpose(transform(imread(imgfile)).flatten()[:, np.newaxis])
                img = transform(imread(imgfile))
                img = np.transpose(img, (2, 0, 1))
                lab = one_hot[int(label)][np.newaxis, :]
                features = np.append(features, img[np.newaxis, :], axis=0)
                labels = np.append(labels, lab, axis=0)
                self.cur_fileline += 1
            self.cur_filepos = mapfile.tell()

        # Check weather reached end of the map file
        if self.cur_fileline >= batch_size:
            self.cur_fileline = self.cur_filepos = 0

        features = features.astype(np.float32)
        labels = labels.astype(np.float32)
        return features, labels


def cnn_batchnorm(features, out_dims):
    "CNN with batch normalization"
    with default_options(activation=relu):
        model = Sequential([
            For(range(4), lambda i: [
                Convolution((5, 5), [16, 32, 32, 64][i], init=glorot_uniform(), pad=True),
                BatchNormalization(map_rank=1),
                MaxPooling((3, 3), strides=(2, 2))
            ]),
            Dense(64, init=glorot_uniform()),
            BatchNormalization(map_rank=1),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(features)


def train_and_evaluate(train_data, test_data, max_epochs, model_func):
    """Train and evaluate function"""
    set_computation_network_trace_level(0)

    # Input variables denoting the features and label data
    input_var = cntk_input((train_data.channels, train_data.height, train_data.width))
    label_var = cntk_input((train_data.classes))

    # apply model to input
    z = model_func(input_var, out_dims=train_data.classes)

    #
    # Training action
    #

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size = train_data.num_samples
    minibatch_size = 128

    # Set training parameters
    lr_per_minibatch = learning_rate_schedule([0.02]*10 + [0.003]*10 + [0.001],
                                              UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight = 0.001

    # trainer object
    learner = momentum_sgd(z.parameters,
                           lr=lr_per_minibatch, momentum=momentum_time_constant,
                           l2_regularization_weight=l2_reg_weight)

    # progress writers
    progress_writers = ProgressPrinter(tag='Training', num_epochs=max_epochs)

    trainer = Trainer(z, (ce, pe), [learner], [progress_writers])

    log_number_of_parameters(z); print()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for _ in range(max_epochs): # loop over epochs
        sample_count = 0
        # loop over minibatches in the epoch
        while sample_count < epoch_size:
            # fetch minibatch and update model with it
            features, labels = train_data.next_minibatch(minibatch_size, rgb_to_he)
            trainer.train_minibatch({input_var : features, label_var : labels})

            # count samples processed so far
            sample_count += features.shape[0]

            # For visualization...
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)

            batch_index += 1
        trainer.summarize_training_progress()

    #
    # Evaluation action
    #
    epoch_size = test_data.num_samples
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer = 0
    metric_denom = 0
    sample_count = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        # Fetch next test min batch.
        current_minibatch = min(minibatch_size, epoch_size - sample_count)
        features, labels = test_data.next_minibatch(minibatch_size, rgb_to_he)

        data = {input_var : features, label_var : labels}
        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += features.shape[0]
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}"
          .format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")

    # Visualize training result:
    window_width = 32
    loss_cumsum = np.cumsum(np.insert(plot_data['loss'], 0, 0))
    error_cumsum = np.cumsum(np.insert(plot_data['error'], 0, 0))

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss'] = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error'] = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width

    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.tight_layout()
    plt.show()

    return softmax(z)

RGB = imread(r"D:\Source\Repos\CNTK_sources\CNTK_Breast\color deconv\test2.jpg")
OD = intensity_to_od(RGB)
RESULT = deconv(OD)
create_txt(RESULT, 1, 'deconv.txt')

# display results
plt.figure(1)
plt.subplot(131)
plt.title("H&E")
plt.imshow(RGB)
plt.subplot(132)
plt.title("Haematoxylin")
plt.imshow(RESULT[:, :, 0], cmap='gray')
plt.subplot(133)
plt.title("Eosin")
plt.imshow(RESULT[:, :, 1], cmap='gray')
plt.show()


ImagSize = 256

# model dimensions
image_height = ImagSize
image_width = ImagSize
num_channels = 2
num_classes = 2

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
DataPath = os.getcwd()
for i in range(2):
    DataPath = os.path.abspath(os.path.join(DataPath, os.pardir))
DataPath = os.path.join(DataPath,'data')

# Change map text files into the CNTK format
print("converting cvedia map to cntk map...", end = '')
nTrain = changeCvediaToCNTKmap(os.path.join(DataPath, 'train.txt'), os.path.join(DataPath, 'train_total_cntk.txt'))
nTest = changeCvediaToCNTKmap(os.path.join(DataPath, 'test.txt'), os.path.join(DataPath, 'test_total_cntk.txt'))
print("finished!")
#print("Number of training samples: {}\nNumber of test samples: {}\n".format(nTrain, nTest))

# Mix map data
print("Mixing the training data...", end='')
MixCNTKmap(os.path.join(DataPath, 'train_total_cntk.txt'), os.path.join(DataPath, 'train_total_cntk_mixed.txt'))
print("finished!")

# Create image readers
train_data = ImageMinibatch(image_width, image_height, num_channels, num_classes, nTrain, os.path.join(DataPath,'train_total_cntk_mixed.txt'))
test_data = ImageMinibatch(image_width, image_height, num_channels, num_classes, nTest, os.path.join(DataPath,'test_total_cntk.txt'))

pred_basic_model_bn = train_and_evaluate(train_data, test_data, max_epochs=10, model_func=cnn_batchnorm)
