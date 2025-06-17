import numpy as np

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Rearranges input images into columns (for convolution).
    
    Parameters
    ----------
    input_data : 4D array (N = #{images}, C = #{channels}, H = height, W = width)
    filter_h : height of the filter
    filter_w : width of the filter
    stride : stridie size
    pad : padding size
    
    Returns
    -------
    col : 2D array of shape (N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # Add zero padding to the input
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # Extract all filter-sized patched from the input
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # Reshape into 2D array
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Reconstructs the origianl image batch from column format.
    
    Parameters
    ----------
    col : 2D array, result of im2col
    input_shape : original shape (N, C, H, W)
    filter_h : height of the filter
    filter_w : width of the filter
    stride : stride size
    pad : padding size
    
    Returns
    -------
    img : 4D array of reconstructed images
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # Reshape col back to patch format
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # Initialize padded output image
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    
    # Accumulate values into image
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # Remove padding and return
    return img[:, :, pad:H + pad, pad:W + pad]