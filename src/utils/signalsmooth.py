"""
signalsmooth.py in GeoPy/src/utils 

From: http://scipy.org/Cookbook/SignalSmooth
"""


# Import modules
import numpy as np


class UnmaskAndPad(object):
    ''' Decorator class to preprocess arrays for smoothing '''
    
    def __init__(self, smoother):
        ''' Store the smoothing operation we are going to apply '''
        self.smoother = smoother

    def __call__(self, data, pad_value=0, **kwargs):
        ''' Unmask and pad data, execute smoother, and restore mask '''
        
        # Check data type
        if not isinstance(data,np.ndarray):
            raise TypeError(data)
        
        # Remove mask
        if isinstance(data, np.ma.masked_array):
            mask = data.mask; fill_value = data._fill_value
            data = data.filled(pad_value) # not actually inplace
            # NOTE: The ".filled" operation returns input as an array with masked data 
            #   replaced by a fill value.
        else:
            mask = None
        
        # Remove NaNs
        if np.issubdtype(data.dtype, np.inexact):
            # NOTE: p.issubdtype(arg1,arg2) returns True if first argument is a typecode 
            #   lower/equal in type hierarchy.
            # NOTE: The above "if" is because integers don't have NaNs.
            nan_mask = np.isnan(data)
            data[nan_mask] = pad_value
            if np.isinf(data).any():
                raise NotImplementedError("Non-finite values except NaN are currently not handled in smoothing.")
        else:
            nan_mask = None
        
        # Apply smoother
        data = self.smoother(data, **kwargs)
        
        # Restore NaNs
        if nan_mask is not None:
            data[nan_mask] = np.NaN
        
        # Restore mask
        if mask is not None:
            data = np.ma.masked_array(data, mask=mask)
            data._fill_value = fill_value
            
        # Return data
        return data              
      

@UnmaskAndPad
def smooth(x, window_len=11, window='hanning'):
    """Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the appropriate size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                (flat window will produce a moving average smoothing)

    Output:
        The smoothed signal
        
    Example:
        import numpy as np    
        t = np.linspace(-2,2,0.1)
        x = np.sin(t)+np.random.randn(len(t))*0.1
        y = smooth(x)
    
    See also: 
    
        numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve  
    """

    # Check if x is 1D
    if x.ndim != 1:
        raise ValueError("Smooth only accepts 1 dimension arrays.")

    # Check window length compared to size of x
    if x.size < window_len:
        raise ValueError("Input vector needs to be of equal size or bigger than window size.")

    # Return x itself if window length is less than 3
    if window_len < 3:
        return x

    # Check window type
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window should be one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # Reflect signal at the ends
    s = np.r_[ 2*x[0]-x[window_len//2:0:-1], x, 2*x[-1]-x[-2:-(window_len//2)-2:-1] ]
    # NOTE: "np.r_" cancatenates arrays together.
        
    # Get w 
    if window == 'flat': 
        w = np.ones(window_len,'d')
        # NOTE: This gives a vector of length window_len and of type float. 
    else:
        w = getattr(np, window)(window_len)
        # NOTE: The getattr() method returns the value of the named attribute of an 
        #   object. If not found, it returns the default value provided to the function.
    
    # Calculate the smoothed signal
    y = np.convolve(w/w.sum(), s, mode='same')
    # NOTE: "w.sum()" here is because we want ot normalize the weight function.
    # NOTE: np.convolve(a,b) = np.convolve(b,a).
    # NOTE: mode='same' returns an output of length max(len(w),len(s)), but here
    #   that becomes equal to len(s).
        
    return y[window_len//2:-(window_len//2)]


from scipy import signal

def twoDim_kern(size, window, sizey=None):
    """ Returns a normalized 2D kernel array for convolutions """
    
    # Sort out sizes
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    
    # Make x and y
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    
    # Make window g
    if window=='gauss':
        g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    elif window=='flat':
        g = np.ones((size,sizey))
    elif window=='hanning':
        g1d_x = np.hanning(size)
        g1d_y = np.hanning(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='hamming':
        g1d_x = np.hamming(size)
        g1d_y = np.hamming(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='bartlett':
        g1d_x = np.bartlett(size)
        g1d_y = np.bartlett(sizey)        
        g = np.sqrt(np.outer(g1d_x,g1d_y))
    elif window=='blackman':    
        g1d_x = np.blackman(size)
        g1d_y = np.blackman(sizey)        
        Temp = np.outer(g1d_x,g1d_y)
        Temp[np.abs(Temp)<1e-15] = 0        
        g = np.sqrt(Temp) 
        # NOTE: For the blackman window some elements have tiny negative values which
        #   become problematic when taking the square root. So I've added the above
        #   code to fix this.

    return g/g.sum()


@UnmaskAndPad
def smooth_image(im, window='gauss', n=10, ny=None):
    """ Blurs the image by convolving with a kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
        
    # Sort out sizes
    n = int(n)
    if not ny:
        ny = n
    else:
        ny = int(ny)
    
    # Make g
    g = twoDim_kern(size=n,window=window,sizey=ny) 
    
    # Reflect signal near edges
    [mx,my] = im.shape
    ox = 2*(n//2)+mx
    oy = 2*(ny//2)+my
    S = np.zeros((ox,oy))
    S[n//2:-(n//2),ny//2:-(ny//2)] = im
    for i in np.arange(n//2,ox-(n//2)):
        S[i,:] = np.r_[ 2*im[i-(n//2),0]-im[i-(n//2),ny//2:0:-1], im[i-(n//2),:], 
            2*im[i-(n//2),-1]-im[i-(n//2),-2:-(ny//2)-2:-1] ]
    for j in np.arange(ny//2,oy-(ny//2)):
        S[:,j] = np.r_[ 2*im[0,j-(ny//2)]-im[n//2:0:-1,j-(ny//2)], 
            im[:,j-(ny//2)], 2*im[-1,j-(ny//2)]-im[-2:-(n//2)-2:-1,j-(ny//2)] ]
    # NOTE: "np.r_" cancatenates arrays together.
    
    # Reflect signal near corners
    TL = np.zeros((n//2,ny//2))
    TR = np.zeros((n//2,ny//2))
    BL = np.zeros((n//2,ny//2))
    BR = np.zeros((n//2,ny//2))
    for i in np.arange(ox-(n//2),ox):
        TL[i-ox+(n//2),:] = 2*S[i,ny//2]-S[i,2*(ny//2):ny//2:-1] 
        TR[i-ox+(n//2),:] = 2*S[i,-1-(ny//2)]-S[i,-2-(ny//2):-2*(ny//2)-2:-1] 
    for i in np.arange(n//2):
        BL[i,:] = 2*S[i,ny//2]-S[i,2*(ny//2):ny//2:-1]
        BR[i,:] = 2*S[i,-1-(ny//2)]-S[i,-2-(ny//2):-2*(ny//2)-2:-1] 
    S[0:n//2,0:ny//2] = BL
    S[ox-(n//2):ox,0:ny//2] = TL
    S[0:n//2,oy-(ny//2):oy] = BR
    S[ox-(n//2):ox,oy-(ny//2):oy] = TR
      
    # Smooth the signal
    improc = signal.convolve(S,g,mode='same')
    
    return(improc[n//2:-(n//2),ny//2:-(ny//2)])


def smooth_demo():
    
    # Import required modules
    import matplotlib.pyplot as plt

    # Make signal
    t = np.linspace(-4,4,100)
    x = np.sin(t)
    xn = x + np.random.randn(len(t)) * 0.1
    
    # Calculate smoothed version of x
    y = smooth(x)
    
    # Window size and types
    ws = 31
    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    
    # Plot the weight functions   
    plt.subplot(211)
    plt.plot(np.ones(ws))
    for w in windows[1:]:
        plt.plot(getattr(np, w)(ws))
    plt.axis([0,30,0,1.1])
    plt.legend(windows)
    plt.title("The smoothing windows")
    
    # Plot the signals and their smoothed versions
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(smooth(xn,window_len=10,window=w))
    l = ['original signal', 'signal with noise']
    l.extend(windows)
    plt.legend(l)
    plt.title("Smoothing a noisy signal")


def smooth_image_demo():
    
    # Import required modules
    import matplotlib.pyplot as plt
    
    # Window types
    windows=['gauss', 'flat', 'hanning', 'hamming', 'bartlett', 'blackman']    
    
    # Make X, Y and Z
    X, Y = np.mgrid[-70:70, -70:70]
    Z = np.cos((X**2+Y**2)/200.)+ np.random.normal(size=X.shape)
    
    # Plot the perturbed signal
    plt.figure()
    plt.subplot(121)
    plt.imshow(Z)
    plt.title("The perturbed signal")
    
    # Plot the kernels and smoothed signals
    for w in windows:
        [n,ny] = Z.shape
        g = twoDim_kern(size=31,window=w)
        Z2 = smooth_image(Z,window=w,n=5)
        plt.figure()
        plt.subplot(121)
        plt.imshow(g) 
        plt.colorbar(orientation="horizontal")
        plt.title("Weight function "+w)
        plt.subplot(122)
        plt.imshow(Z2)  
        plt.colorbar(orientation="horizontal")
        plt.title("Smoothed using window "+w)
        

# Main program        
if __name__=='__main__':
    
    # Import modules
    import matplotlib.pyplot as plt
    
    # Part 1: 1D
    smooth_demo()
    
    # Part 2: 2D
    smooth_image_demo()
    
