from common import find_mxnet
import mxnet as mx
import numpy as np
import urllib
import matplotlib.pyplot as plt
import csv
import json
import codecs

def swap(data):
    """
    reshape to 2nd and 4th axes
    """
    res = data.swapaxes(2, 3)
    res = res.swapaxes(1, 2)
    res = res.astype(np.float32)/255
    return res

def transform(data):
    #data = mx.image.imresize(data, 32, 32)
    res = data.transpose((2,0,1))
    #data = mx.nd.swapaxes(data, 0, 2)
    res = res.astype(np.float32)
    return res

def get_model(prefix, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (128,3,32,32))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params)
    return mod

def get_val_iter():
    cifar10_val = mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform)
    swappedval = swap(cifar10_val._data)
    val = mx.io.NDArrayIter(
        swappedval, cifar10_val._label, 128)
    return val

def get_groundtruth():
    cifar10_val = mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform)
    return cifar10_val._label

def get_accuracy(groundtruth, prefix, i):
    cifar_model = get_model(prefix, i)
    print("load model",i,"successfully")
    val_iter = get_val_iter()
    predictions = cifar_model.predict(val_iter)

    predicted_label = predictions.asnumpy().argmax(1)
    correct = np.sum(predicted_label == groundtruth)
    accuracy = float(correct)/len(groundtruth)
    print(accuracy)

    return accuracy

def plot(alldata, path, workernum):
    # plot
    accuracies = []
    batchs = []
    runtime = []
    print(type(alldata))
    for key,value in alldata.items():
        accuracies.append(value['accuracy'])
        runtime.append(value['runtime'])
        batchs.append(key*50)
    # Runtime-Acuuracy Plot
    plt.figure()
    plt.plot(runtime,accuracies,label="KRum Runtime-Acuuracy Plot with "+ workernum+" workers")
    plt.legend()
    plt.xlabel('Runtime (minutes)')
    plt.ylabel('Validation Accuracy (%)')
    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,
                        bottom=0.13, top=0.91)
    plt.savefig(path+"Runtime-Acuuracy.jpg",dpi=600)
    # Batch-Acuuracy Plot
    plt.figure()
    plt.plot(batchs,accuracies,label="KRum Batch-Acuuracy Plot with "+ workernum+" workers")
    plt.legend()
    plt.xlabel('Batch')
    plt.ylabel('Validation Accuracy (%)')
    plt.subplots_adjust(left=0.18, wspace=0.25, hspace=0.25,
                        bottom=0.13, top=0.91)
    plt.savefig(path+"Batch-Acuuracy.jpg",dpi=600)

def readlog(path):
    return 0

if __name__ == '__main__':
    logroot = '/Users/zhuqishan/Desktop/DML-research/data/logs/models/'
    paramsroot = '/Users/zhuqishan/Desktop/DML-research/data/params/models/'
    figureroot = '/Users/zhuqishan/Desktop/DML-research/data/figures/'

    algo = 'krum/' # 'tmean/', 'optmean/'
    workernum = '4/' # '4/', '1/'


    logpath = logroot+algo+workernum+'1failure.log'

    # rows = []

    groundtruth = get_groundtruth()
    print("get groundtruth successfully")

     # '1':{'runtime':%d, 'accuracy':%f, params_prefix:%s}
    alldata = {}

    with open(logpath, 'r+') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ' ')
        for row in csvreader:
            if (len(row) > 3 and row[3] == 'Saved'):
#                rows.append(row)
                cpnum = (int)(row[0].split('[')[1].split(']')[0])
                if cpnum not in alldata:
                    datadict = {}
                    datadict['runtime'] = float(row[1].split(':')[1])
                    datadict['prefix'] = row[-1].split('/')[-1].split('-')[0]
                    parampath = paramsroot +algo+workernum+ datadict['prefix']
                    accuracy = get_accuracy(groundtruth, parampath, cpnum)
                    datadict['accuracy'] = accuracy
                    alldata[cpnum] = datadict

    # save data
    jsdata = json.dumps(alldata)
    file = open(figureroot+algo+workernum+'data.json','w+')
    file.write(jsdata)
    file.close()


    # plot data
    # plot(alldata, figureroot+algo+workernum, workernum)
