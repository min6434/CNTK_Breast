# Import the relevant components
from __future__ import print_function

from CNNFunctions import *

ImagSize = 256    
Job_ID = 'SygDNXzjqQxPAWC2A7Pes3L2m9EBY2dJ' # Job ID obtained by the cvedia-cli 

# model dimensions
image_height = ImagSize
image_width  = ImagSize
num_channels = 3
num_classes  = 2

DataPath = os.getcwd()
for i in range(2):
    DataPath = os.path.abspath(os.path.join(DataPath, os.pardir))
DataPath = os.path.join(DataPath,'data')
#DataPath = os.path.join(DataPath,Job_ID)

# Change map text files into the CNTK format
print("converting cvedia map to cntk map...", end = '')
nTrain = changeCvediaToCNTKmap(os.path.join(DataPath,'train.txt'), os.path.join(DataPath,'train_total_cntk.txt'))
nTest = changeCvediaToCNTKmap(os.path.join(DataPath,'test.txt'), os.path.join(DataPath,'test_total_cntk.txt'))
print("finished!")
print("Number of training samples: {}\nNumber of test samples: {}\n".format(nTrain, nTest))

# Calculate average pixel data and put them into the XML for CNTK
print("calculating an average image...", end = '')
meanImg = saveMean(os.path.join(DataPath,'train_total_cntk.txt'), image_height, image_width, num_channels, nTrain)
saveMeanXML(os.path.join(DataPath,'breast_mean.xml'), meanImg, ImagSize)
print("finished!")

# Mix map data
print("Mixing the training data...", end = '')
MixCNTKmap(os.path.join(DataPath,'train_total_cntk.txt'), os.path.join(DataPath,'train_total_cntk_mixed.txt'))
print("finished!")

# Create image readers
reader_train = create_reader(os.path.join(DataPath,'train_total_cntk_mixed.txt'), os.path.join(DataPath,'breast_mean.xml'), image_width, image_height, num_channels, num_classes, True)
reader_test  = create_reader(os.path.join(DataPath,'test_total_cntk.txt'), os.path.join(DataPath,'breast_mean.xml'), image_width, image_height, num_channels, num_classes, False)

pred_basic_model_bn = train_and_evaluate(reader_train, reader_test, image_width, image_height, num_channels, num_classes,\
    nTrain, nTest, max_epochs=10, model_func=create_basic_model_with_batch_normalization)

label_lookup = ["healty tissue", "metastases"]
nTotal = 0
nFalse = 0
for line in open(os.path.join(DataPath,'test_total_cntk.txt'), 'r'):
    imgFile, label = line.split('\t')
    result = eval(pred_basic_model_bn, imgFile, meanImg)
    nTotal += 1
    if result != int(label):
        print("real value: ", label_lookup[int(label)], end = ", ")
        print("network result: ", label_lookup[result])
        nFalse += 1

print( "Accuracy {}%".format( (nTotal-nFalse)/nTotal*100 ) )  