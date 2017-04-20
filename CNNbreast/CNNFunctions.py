# Functions for breast cancer classification using CNN
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.misc import imread, imresize

import xml.etree.cElementTree as et
import xml.dom.minidom

import cntk as C
from cntk.layers import default_options, Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense, Sequential, For
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms 
from cntk.initializer import glorot_uniform, he_normal
from cntk import Trainer
from cntk.learners import momentum_sgd, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule
from cntk import cross_entropy_with_softmax, classification_error, relu, input, softmax, element_times
from cntk.logging import *

# Use CPU in test environment.
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.set_default_device(C.device.cpu())
    else:
        C.device.set_default_device(C.device.gpu(0))


#####################################################################################################################################################
# Get average pixels of the images
def saveMean(inputFile):
    "Save mean value of the train images"
    meanImg = np.zeros((3,256,256))
    total = 0
    for i, line in enumerate(open(inputFile, 'r')):
        imgFile, label = line.split('\t')
        img = imread(imgFile)
        
        if img.shape[0] != 256 or img.shape[1] != 256:
            img = imresize(img, (256, 256))
        
        img = np.swapaxes(img,0,2)
        meanImg += img
        
    meanImg = np.divide(meanImg, i+1)
    return meanImg


#####################################################################################################################################################
# Create mean XML for CNTK
def saveMeanXML(fname, data, imgSize):
    "Create a XML flie of the mean image data for CNTK format" 
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = \
    ' '.join(['%e\n' % n if (i+1)%4 == 0 else '%e' % n for i, n in enumerate(np.reshape(data, (imgSize * imgSize * 3)))]) # long txt cannot be pared by CNTK paser

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))


#####################################################################################################################################################
# Change cvedia format map file into the CNTK format map file
def changeCvediaToCNTKmap(inputFile, outputFile, dataPath):
    "Change cvedia format map file into the CNTK format map file"
    with open(outputFile, 'w') as CNTKFile: 
        for line in open(inputFile, 'r'):
            line = line.replace(' ', '\t') # CNTK paser column using tab
            line = line.replace('/', '\\') # linux path uses / and winows uses \
            line = line.replace('.\\SygDNXzjqQxPAWC2A7Pes3L2m9EBY2dJ', dataPath)
            CNTKFile.write(line)


#####################################################################################################################################################
# Define the reader for both training and evaluation action.
def create_reader(map_file, mean_file, image_width, image_height, num_channels, num_classes, train):
    "Image reader creation for the learner"
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("This tutorials depends 201A tutorials, please run 201A first.")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8) # train uses data augmentation (translation only)
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))


#####################################################################################################################################################
def create_basic_model_terse(input, out_dims):

    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((5,5), [32,32,64][i], init=glorot_uniform(), pad=True),
                MaxPooling((3,3), strides=(2,2))
            ]),
            Dense(64, init=glorot_uniform()),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)


#####################################################################################################################################################
def create_basic_model_with_dropout(input, out_dims):

    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((5,5), [32,32,64][i], init=glorot_uniform(), pad=True),
                MaxPooling((3,3), strides=(2,2))
            ]),
            Dense(64, init=glorot_uniform()),
            Dropout(0.25),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)


#####################################################################################################################################################
def create_basic_model_with_batch_normalization(input, out_dims):

    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((5,5), [32,32,64][i], init=glorot_uniform(), pad=True),
                BatchNormalization(map_rank=1),
                MaxPooling((3,3), strides=(2,2))
            ]),
            Dense(64, init=glorot_uniform()),
            BatchNormalization(map_rank=1),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)



#####################################################################################################################################################
def train_and_evaluate(reader_train, reader_test, image_width, image_height, num_channels, num_classes, max_epochs, model_func):
    # Input variables denoting the features and label data
    input_var = C.input((num_channels, image_height, image_width))
    label_var = C.input((num_classes))
    
    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = element_times(feature_scale, input_var)
    
    # apply model to input
    z = model_func(input_var_norm, out_dims=2)

    #
    # Training action
    #

    # loss and metric
    ce = cross_entropy_with_softmax(z, label_var)
    pe = classification_error(z, label_var)

    # training config
    epoch_size     =  461
    minibatch_size = 64

    # Set training parameters
    lr_per_minibatch       = learning_rate_schedule([0.01]*10 + [0.003]*10 + [0.001], UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.001
    
    # trainer object
    learner = momentum_sgd(z.parameters, 
                           lr = lr_per_minibatch, momentum = momentum_time_constant, 
                           l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    trainer = Trainer(z, (ce, pe), [learner], [progress_printer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far
            
            # For visualization...            
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
            
            batch_index += 1
        trainer.summarize_training_progress()
        
    #
    # Evaluation action
    #
    epoch_size     = 64
    minibatch_size = 64

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")
    
    # Visualize training result:
    window_width            = 16
    loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

    # Moving average.
    plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss ')

    plt.show()

    plt.subplot(212)
    plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error ')
    plt.show()
    
    return softmax(z)


import PIL

def eval(pred_op, image_path):
    label_lookup = ["healty tissue", "metastases"]
    image_mean   = 133.0
    image_data   = np.array(PIL.Image.open(image_path), dtype=np.float32)
    
    if image_data.shape[0] != 256 or image_data.shape[1] != 256:
        image_data = imresize(image_data, (256, 256))
    
    image_data = image_data.astype(dtype = np.float32)

    image_data  -= image_mean
    image_data   = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    
    result       = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))
    
    # Return top 3 results:
    top_count = 2
    result_indices = (-np.array(result)).argsort()[:top_count]

    #print("Top 2 predictions:")
    #for i in range(top_count):
    #    print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100))
    
    return result_indices[0]