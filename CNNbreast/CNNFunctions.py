"""Functions for breast cancer classification using CNN""" 
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
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
from cntk import cross_entropy_with_softmax, classification_error, relu, input, softmax, element_times, reduce_mean
from cntk.logging import *

from cntk.debugging import set_computation_network_trace_level
from cntk.debugging import *

# Use CPU in test environment.
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.set_default_device(C.device.cpu())
        print('CPU!')
    else:
        C.device.set_default_device(C.device.gpu(0))
        print('GPU!')

def saveMean(inputFile, image_height, image_width, num_channels, num_samples):
    "Save mean value of the train images"
    meanImg = np.zeros((num_channels, image_height, image_width), dtype=np.float32)

    for line in open(inputFile, 'r'):
        imgFile, label = line.split('\t')
        img = imread(imgFile)
        if img.shape[0] != image_width or img.shape[1] != image_height:
            img = imresize(img, (image_width, image_height))

        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        #img = np.swapaxes(img,0,2)

        meanImg += img

    meanImg = np.divide(meanImg, num_samples)

    return meanImg


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
        f.write(x.toprettyxml(indent='  '))


# Change cvedia format map file into the CNTK format map file
def changeCvediaToCNTKmap(inputFile, outputFile):
    NumSamples = 0
    NumHealthy = 0
    NumTumor = 0

    "Change cvedia format map file into the CNTK format map file"
    with open(outputFile, 'w') as CNTKFile:
        for line in open(inputFile, 'r'):
            line = line.replace(' ', '\t') # CNTK paser column using tab
            line = line.replace('/', '\\') # linux path uses / and winows uses
            line = line.replace('.\\', '.\\..\\..\\data\\')

            imgFile, label = line.split('\t')
            img = imread(imgFile)

            if img.shape[0] == 256 and img.shape[1] == 256:
                CNTKFile.write(line)
                NumSamples += 1
                if int(label) == 0: NumHealthy += 1
                else: NumTumor += 1

    print("\nHealty tissue: {} Tumor tissue: {}".format(NumHealthy, NumTumor))
    return NumSamples


# Mix label data randomly
import random
def MixCNTKmap(inputFile, outputFile):
    "Mix lines in the cntk map randomly"
    with open(inputFile, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(outputFile, 'w') as target:
        for _, line in data:
            target.write(line)

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
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear')#,
        #xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features=StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels=StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))


# basic convolutional network
def create_basic_model_terse(input, out_dims):
    ""
    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((3, 3), [16, 32, 64][i], init=glorot_uniform(), pad=True),
                Convolution((3, 3), [16, 32, 64][i], init=glorot_uniform(), pad=True),
                MaxPooling((3, 3), strides=(2, 2))
            ]),
            Dense(64, init=glorot_uniform()),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)


def create_basic_model_with_dropout(input, out_dims):
    ""
    with default_options(activation=relu):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((5, 5), [32, 32, 64][i], init=glorot_uniform(), pad=True),
                MaxPooling((3, 3), strides=(2, 2))
            ]),
            Dense(64, init=glorot_uniform()),
            Dropout(0.25),
            Dense(out_dims, init=glorot_uniform(), activation=None)
        ])

    return model(input)


def create_basic_model_with_batch_normalization(input, out_dims):
    ""
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

    return model(input)




from cntk.ops import combine, times, element_times, AVG_POOLING


def convolution_bn(input, filter_size, num_filters, strides=(1, 1), init=he_normal(), activation=relu):
    ""
    if activation is None:
        activation = lambda x: x

    r = Convolution(filter_size, num_filters, strides=strides, init=init, activation=None, pad=True, bias=False)(input)
    r = BatchNormalization(map_rank=1)(r)
    r = activation(r)

    return r



def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters)
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)
    p = c2 + input
    return relu(p)



def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3, 3), num_filters, strides=(2, 2))
    c2 = convolution_bn(c1, (3, 3), num_filters, activation=None)

    s = convolution_bn(input, (1, 1), num_filters, strides=(2, 2), activation=None)

    p = c2 + s
    return relu(p)



def resnet_basic_stack(input, num_filters, num_stack):
    assert (num_stack>0)

    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r



def create_resnet_model(input, out_dims):
    conv = convolution_bn(input, (3,3), 16)
    r1_1 = resnet_basic_stack(conv, 16, 3)

    r2_1 = resnet_basic_inc(r1_1, 32)
    r2_2 = resnet_basic_stack(r2_1, 32, 2)

    #r3_1 = resnet_basic_inc(r2_2, 64)
    #r3_2 = resnet_basic_stack(r3_1, 64, 2)
    r3_1 = resnet_basic_inc(r2_2, 32)
    r3_2 = resnet_basic_stack(r3_1, 32, 2)

    # Global average pooling
    pool = AveragePooling(filter_shape=(8,8), strides=(1,1))(r3_2)    
    net = Dense(out_dims, init=he_normal(), activation=None)(pool)
    
    return net

def train_and_evaluate(reader_train, reader_test, image_width, image_height, num_channels, num_classes, num_train, num_test, max_epochs, model_func):
   
    set_computation_network_trace_level(0)
    
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
    epoch_size     =  num_train
    minibatch_size = 128

    # Set training parameters
    lr_per_minibatch       = learning_rate_schedule([0.02]*10 + [0.003]*10 + [0.001], UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.001
    
    # trainer object
    learner = momentum_sgd(z.parameters, 
                           lr = lr_per_minibatch, momentum = momentum_time_constant, 
                           l2_regularization_weight=l2_reg_weight)
    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)

    # progress writers
    progress_writers = ProgressPrinter(tag='Training', num_epochs=max_epochs)
    tensorboard_logdir= 'log'
    tensorboard_writer = None
    if tensorboard_logdir is not None:
        tensorboard_writer = TensorBoardProgressWriter(freq=10, log_dir=tensorboard_logdir, model=z)
        #progress_writers.append(tensorboard_writer)
   
    trainer = Trainer(z, (ce, pe), [learner], [progress_writers, tensorboard_writer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()

    # perform model training
    profiler_dir = 'profiler'
    if profiler_dir:
        start_profiler(profiler_dir, True)

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
        
        # Log mean of each parameter tensor, so that we can confirm that the parameters change indeed.
        if tensorboard_writer:
            for parameter in z.parameters:
                tensorboard_writer.write_value(parameter.uid + "/mean", reduce_mean(parameter).eval(), epoch)

        model_dir = ''
        if model_dir:
            z.save(os.path.join(model_dir, network_name + "_{}.dnn".format(epoch)))
        enable_profiler() # begin to collect profiler data after first epoch

    if profiler_dir:
        stop_profiler()
        
    #
    # Evaluation action
    #
    epoch_size     = num_test
    minibatch_size = 16

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


import PIL
def eval(pred_op, image_path, image_mean):
    label_lookup = ["healty tissue", "metastases"]
    image_data = cv.imread(image_path)
    image_data = np.array(image_data, dtype=np.float32)

    if image_data.shape[0] != 256 or image_data.shape[1] != 256:
        image_data = cv.resize(image_data, (256, 256))

    image_data = image_data.astype(dtype=np.float32)

    image_data = np.transpose(image_data, (2, 0, 1))
    #image_data = np.swapaxes(image_data,0,2)
    #image_data = np.swapaxes(image_data,1,2)
    #image_data -= image_mean[:,:,::-1]#cv.cvtColor(image_mean, cv.COLOR_RGB2BGR)

    image_data = np.ascontiguousarray(image_data, dtype=np.float32)
    result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))

    # Return top 3 results:
    top_count = 2
    result_indices = (-np.array(result)).argsort()[:top_count]

    #print("Top 2 predictions:")
    #for i in range(top_count):
    #    print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100))

    return result_indices[0]