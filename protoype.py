import cv2
import numpy as np
import matplotlib.pyplot as plt
import random as rng

''' Distortion Functions'''
# - Start - #
def get_transform_matrix(src, dst):
    return cv2.getPerspectiveTransform(src, dst)

def warp(img, matrix, img_shape):
    return cv2.warpPerspective(img, matrix, img_shape)

def get_unwarp_matrix(dst, src):
    return cv2.getPerspectiveTransform(dst, src)
# - End - #

''' Useful Functions '''
# - Start - #

def weighted_img(img, initial_img, α=0.5, β=1, γ=1):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def nothing(x):
    pass

def display_images(images, c=4, r=4):
    w = 20
    h = 20
    fig = plt.figure(figsize=(100, 100))
    columns = c
    rows = r

    for i in range(1, len(images) + 1):
        img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def call_with_trackbars(parameters, variables, callback):
    cv2.namedWindow('Variables', cv2.WINDOW_NORMAL)

    for variable in variables:
        cv2.createTrackbar(variable['name'], 'Variables', variable['default'], variable['max'], nothing)

    cv2.resizeWindow('Variables', 500, 200)
    cv2.moveWindow('Variables', 0, 500)  # Doesn't work on macOS at the moment due to a bug in OpenCV (https://github.com/opencv/opencv/issues/16343)

    values = {}
    last_values = None
    while (1):
        # Break when the "Escape" key is pressed.
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        for variable in variables:
            values[variable['name']] = cv2.getTrackbarPos(variable['name'], 'Variables')

        if (values == last_values):
            continue
        last_values = values.copy()

        print(values)
        callback({**parameters, **values})

def colorize(dist):
    # stretch to full dynamic range
    stretch = skimage.exposure.rescale_intensity(dist, in_range='image', out_range=(0, 255)).astype(np.uint8)

    '''
    oldMax = float(dist.max())
    oldMin = float(dist.min())
    newMax = 255.0
    newMin = 0.0
    oldRange = (oldMax - oldMin)
    stretch = np.zeros_like(dist)

    dist = np.clip(dist, oldMin, oldMax)
    h, w = dist.shape[:2]

    dist = (dist - oldMin) / (oldMax - oldMin)
    stretch = np.asarray(dist * (newMax - newMin) + newMin, dtype=np.uint8)

    cv2.imshow("stretch", stretch)
    cv2.waitKey(0)

    #print(stretch)
    '''
    #convert to 3 channels
    stretch = cv2.merge([stretch, stretch, stretch])


    # define colors
    color1 = (0, 0, 255)    # red
    color2 = (0, 165, 255)  # orange
    color3 = (0, 255, 255)  # yellow
    color4 = (255, 255, 0)  # cyan
    color5 = (255, 0, 0)    # blue
    color6 = (128, 64, 64)  #violet
    colorArr = np.array([[color1, color2, color3, color4, color5, color6]], dtype=np.uint8)

    #resize lut to 256 (or more) values. This creates a single line of color
    lut = cv2.resize(colorArr, (256, 1), interpolation=cv2.INTER_LINEAR)

    # apply lut
    result = cv2.LUT(stretch, lut)
    return result

# - End - #

''' Sobel Block '''
# - Start - #

def sobel(img, xl, xh, yl, yh, ml, mh, dl, dh):
    def apply_thresh(img, threshold=(0, 255)):
        '''
        applies thresholds to input 8-bit image
        :param img: input image
        :param threshold: threshold to be applied, must be a tuple of 8-bit values
        :return: resulting mask
        '''
        # scale image to 8 bit
        img_uint8 = np.uint8(255 * img / np.max(img))
        thresh_mask = np.zeros_like(img_uint8)
        thresh_mask[(img_uint8 >= threshold[0]) & (img_uint8 <= threshold[1])] = 1
        return thresh_mask

    def sobel_xy_gradients(img, sobel_ksize):
        '''
        apply both x and y edge detection through sobel transform to image
        :param img: input image
        :param sobel_ksize: kernel size for sobel transform
        :return: returns detected edges in two images, x and y
        '''
        x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
        y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
        return x, y

    def sobel_absolute_value_mask(image, axis='x', sobel_ksize=3, threshold=(0, 255)):
        '''
        applies absolute value component of sobel edge detection and creates a resulting mask
        according to the threshold
        :param image: input image
        :param axis: character input axis to apply the edge detection to in either x
                     or y direction
        :param sobel_ksize: kernel size for sobel transform
        :param threshold: tuple of two values in range of 0 to 255
        :return: threshold-applied mask image
        '''
        if axis == 'x':
            sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_ksize))
        elif axis == 'y':
            sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_ksize))
        else:
            raise "axis must be specified as x or y value"
        mask = apply_thresh(sobel, threshold)
        return mask

    def sobel_magnitude_mask(img, sobel_ksize=3, threshold=(0, 255)):
        '''
        applies magnitude component of sobel edge detection and creates a resulting mask
        according to the threshold
        :param img: input image
        :param sobel_ksize: kernel size for sobel transform
        :param threshold: tuple of two values in range of 0 to 255
        :return: threshold-applied mask image
        '''
        x, y = sobel_xy_gradients(img, sobel_ksize)

        sobel_magnitude = np.sqrt(x ** 2 + y ** 2)
        mask = apply_thresh(sobel_magnitude, threshold)
        return mask

    def sobel_direction_mask(img, sobel_ksize=3, threshold=(0, 255)):
        '''
        applies direction component of sobel edge detection and creates a resulting mask
        according to the threshold
        :param img: input image
        :param sobel_ksize: kernel size for sobel transform
        :param threshold: tuple of two values in range of 0 to 255
        :return: threshold-applied mask image
        '''
        x, y = sobel_xy_gradients(img, sobel_ksize)
        sobel_direction = np.arctan2(np.absolute(y), np.absolute(x))
        mask = apply_thresh(sobel_direction, threshold=threshold)
        return mask

    def sobel_mask(img, xl, xh, yl, yh, ml, mh, dl, dh):
        '''
        applies three different components of sobel edge detection with individual threshold
        values for each.
        :param img: input image
        :return: mask of edge-detected image
        '''
        sobel_x = sobel_absolute_value_mask(img, axis='x', sobel_ksize=3, threshold=(xl, xh))
        sobel_y = sobel_absolute_value_mask(img, axis='y', sobel_ksize=3, threshold=(yl, yh))
        magnitude = sobel_magnitude_mask(img, sobel_ksize=3, threshold=(ml, mh))
        direction = sobel_direction_mask(img, sobel_ksize=3, threshold=(dl, dh))

        s_mask = np.zeros_like(img)
        s_mask[((sobel_x == 1) & (sobel_y == 1)) |
               ((magnitude == 1) & (direction == 1))] = 1

        return s_mask

    def hls(img):
        '''
        transforms image to the HLS colorspace from RGB
        :param img: input image, must be in RGB for correct conversion
        :return: image in HLS colorspace
        '''
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    def color_mask(img):
        '''
        creates a color mask according to the internal threshold inside this
        function
        :param img: input image
        :return: mask of threshold-applied image
        '''
        # c_mask = apply_thresh(img, threshold=(170, 255))
        c_mask = np.zeros_like(img)
        c_mask[(img >= 170) & (img <= 255)] = 1
        return c_mask

    def edges_mask(img, xl, xh, yl, yh, ml, mh, dl, dh):
        '''
        function created for better organization
        :param img: input image
        :return: color mas and edges mask
        '''
        c_mask = color_mask(img)
        s_mask = sobel_mask(img, xl, xh, yl, yh, ml, mh, dl, dh)
        return s_mask

    def find_mask(img, xl, xh, yl, yh,ml, mh, dl, dh):
        '''
        function will apply HLS transform, create masks for edges and color, then
        combine them into one mask.
        :param img: input image
        :return: stacked mask of edges and colors
        '''
        s_mask = edges_mask(img, xl, xh, yl, yh,ml, mh, dl, dh)
        mask = np.zeros_like(s_mask)
        mask[(s_mask == 1)] = 1
        return mask

    return find_mask(img, xl, xh, yl, yh, ml, mh, dl, dh)

# - End - #

''' Trackbar Algorithms'''
# - Start - #
def prototypeAlg(p):

    # get parameters from p 
    img = p['img']
    h, w = img.shape[:2]
    
    # display setup 
    stack1 = None
    stack2 = None
    stack3 = None
    stack4 = None
    #out1 = np.dstack((stack1, stack1, stack1))
    #out2 = np.dstack((stack2, stack2, stack2))
    #out3 = np.dstack((stack3, stack3, stack3))
    #out4 = np.dstack((stack4, stack4, stack4))
    
    out = None
    #out = np.hstack((out1, out2, out3, out4))

    cv2.imshow('Variables', out)
    return 

# - End - #


def main():
    images = []
    imgPaths = []
    imgPaths.append("")
    img = cv2.imread(imgPaths[0])

    h, w = img.shape[:2]


    ''' Trackbar parameters'''
    variables = [
        {
            'name': '',
            'default': 0,
            'max': 255
        },
        {
            'name': '',
            'default': 0,
            'max': 255
        },
    ]

    # launch trackbars callback function
    cv2.destroyAllWindows()
    call_with_trackbars({ 'img': img }, variables, prototypeAlg)
    

    # End main()

if __name__ == '__main__':
    main()
    