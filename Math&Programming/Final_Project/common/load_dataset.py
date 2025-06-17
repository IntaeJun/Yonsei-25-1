import numpy as np

def load_mnist(flatten=True, normalize=False, one_hot_label=False):
    from tensorflow.keras.datasets import mnist

    # 1) Load the MNIST dataset
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    #    x_*: (60000, 28, 28) or (10000, 28, 28), dtype=uint8
    #    t_*: (60000,) or (10000,), values 0–9

    # 2) Normalize pixel values to [0,1] if requested
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test  = x_test.astype(np.float32) / 255.0
    else:
        x_train = x_train.astype(np.uint8)
        x_test  = x_test.astype(np.uint8)

    # 3) Flatten the images to shape (N, 784) if requested
    if flatten:
        x_train = x_train.reshape(-1, 28 * 28)
        x_test  = x_test.reshape(-1, 28 * 28)
    else:
        x_train = x_train.reshape(-1, 1, 28, 28)       # (N, 1, 28, 28)
        x_test  = x_test.reshape(-1, 1, 28, 28)

    # 4) Convert labels to one-hot encoding if requested
    if one_hot_label:
        def _to_one_hot(labels, num_classes):
            """Convert integer labels to one-hot encoded format."""
            return np.eye(num_classes)[labels]
        
        t_train = _to_one_hot(t_train, num_classes=10)
        t_test  = _to_one_hot(t_test,  num_classes=10)

    return (x_train, t_train), (x_test, t_test)

def load_cifar10(flatten=True, normalize=False, one_hot_label=False):
    from tensorflow.keras.datasets import cifar10

    # 1) Load the CIFAR-10 dataset
    (x_train, t_train), (x_test, t_test) = cifar10.load_data()
    #    x_*: (50000, 32, 32, 3), dtype=uint8
    #    t_*: (50000, 1), values 0–9
    
    # 2) Normalize pixel values to [0,1] if requested
    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test  = x_test.astype(np.float32) / 255.0
    else:
        x_train = x_train.astype(np.uint8)
        x_test  = x_test.astype(np.uint8)

    # 3) Flatten the images to shape (N, 3072) if requested
    if flatten:
        x_train = x_train.reshape(-1, 32 * 32 * 3)
        x_test  = x_test.reshape(-1, 32 * 32 * 3)
    else:
        x_train = x_train.transpose(0, 3, 1, 2)  # 🟡 Convert to (N, C, H, W)
        x_test  = x_test.transpose(0, 3, 1, 2)

    # 4) Convert labels to one-hot encoding if requested
    if one_hot_label:
        def _to_one_hot(labels, num_classes):
            return np.eye(num_classes)[labels.flatten()]
        t_train = _to_one_hot(t_train, num_classes=10)
        t_test  = _to_one_hot(t_test,  num_classes=10)

    return (x_train, t_train), (x_test, t_test)