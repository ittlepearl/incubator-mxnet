from common import find_mxnet
import mxnet as mx
import urllib

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
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
    mod = mx.mod.Module(symbol=sym, context=mx.cpu())
    mod.bind(for_training=False, data_shapes=[('data', (128,3,32,32))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params)
    return mod

def get_val_iter():
    cifar10_val = mx.gluon.data.vision.CIFAR10(root='~/.mxnet/datasets/cifar10', train=False, transform=transform)
    swappedval = swap(cifar10_val._data)
    val = mx.io.NDArrayIter(
        swappedval, val_cifar10._label, 128)
    return val

def main():

    rootpath = '/home/ubuntu/qishanz2/src/incubator-mxnet/example/image-classification/'
    prefix = './model/lenet'


    mod = get_model(prefix, 0)

    val_iter = get_val_iter()

    predictions = cifar_model.predict(val_iter)

    print(predictions.shape())
    predicted_label = predictions[4].asnumpy().argmax()

if __name__ == '__main__':
    main()
