#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.path as mplPath
import numpy as np
import cv2
# % matplotlib inline

last_slope_left = 1
last_slope_right = 1


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


import math

def grayscale(img):
    
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    
    
    #[slopeSum, count, yIntersectSum, x, y]
    leftLane = [0, 0, 0, 0, 0]
    rightLane = [0, 0, 0, 0, 0]
    
    # Image dimentions (y, x): 540, 960 
    
    # used to find if a point is within the polygon
    # https://stackoverflow.com/questions/39660851/deciding-if-a-point-is-inside-a-polygon-python
    vertices = np.array([[100,540],[420, 330], [520, 330], [900,540]])
    bbPath = mplPath.Path(vertices)
    
    minX = 100
    maxX = 900
    
    minY = 330
    maxY = 540
    
    # Go through each line and keep track of the slope and y-intersect of each line
    # we will average these out later to obtain the lane marking.
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            # Only calculate slope if it's within our bounded region
            if (bbPath.contains_point([x1, y1]) is not True):
                continue
                
            m = (y2 - y1) / (x2 - x1)
            #b = y - mx
            b = y1 - (m * x1)
            if (m > 0):
                rightLane[0] = rightLane[0] + m
                rightLane[1] += 1
                rightLane[2] += b
                rightLane[3] = x1
                rightLane[4] = y1
            else:
                leftLane[0] = leftLane[0] + m
                leftLane[1] += 1
                leftLane[2] += b
                leftLane[3] = x1
                leftLane[4] = y1
            
            #cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 2)
    
    left_lane_slope = 0
    left_lane_y_intercept = 0
    right_lane_slope = 0
    right_lane_y_intercept = 0
    
    global last_slope_left
    global last_slope_right
    
    if (leftLane[1] != 0):
        left_lane_slope = leftLane[0] / leftLane[1]
        left_lane_y_intercept = leftLane[2] / leftLane[1]
        last_slope_left = left_lane_slope
    else:
        left_lane_slope = last_slope_left
    
    if (rightLane[1] != 0):
        right_lane_slope = rightLane[0] / rightLane[1]
        right_lane_y_intercept = rightLane[2] / rightLane[1]
        last_slope_right = right_lane_slope
    else:
        right_lane_slope = last_slope_right
    
    
    # x = (y - b) / m
    # y = mx + b
    x_bottom_left_lane = (maxY - left_lane_y_intercept) / left_lane_slope
    y_bottom_left_lane = left_lane_slope * x_bottom_left_lane + left_lane_y_intercept

    
    x_top_left_lane = (minY - left_lane_y_intercept) / left_lane_slope
    y_top_left_lane = left_lane_slope * x_top_left_lane + left_lane_y_intercept
    
    cv2.line(img, (int(round(x_bottom_left_lane)), int(round(y_bottom_left_lane))), (int(round(x_top_left_lane)), int(round(y_top_left_lane))), [255, 0, 0], thickness)
    
    
    minX_right = 100
    maxX_right = 900
    
    minY_right = 330
    maxY_right = 540
    
    x_bottom_right_lane = (maxY_right - right_lane_y_intercept) / right_lane_slope
    y_bottom_right_lane = right_lane_slope * x_bottom_right_lane + right_lane_y_intercept
    
    x_top_right_lane = (minY - right_lane_y_intercept) / right_lane_slope
    y_top_right_lane = right_lane_slope * x_top_right_lane + right_lane_y_intercept
    
    cv2.line(img, (int(round(x_bottom_right_lane)), int(round(y_bottom_right_lane))), (int(round(x_top_right_lane)), int(round(y_top_right_lane))), [0, 255, 0], thickness)
    
    
    # DEBUG
    # DRAW the area of interes
    # vertices = np.array([[100,540],[420, 330], [520, 330], [900,540]])
    #cv2.line(img, (100, 540), (420, 330), [0, 0, 255], 5)
    #cv2.line(img, (420, 330), (520, 330), [0, 0, 255], 5)
    #cv2.line(img, (520, 330), (900, 540), [0, 0, 255], 5)
    #cv2.line(img, (900, 540), (100, 540), [0, 0, 255], 5)
        


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, [0, 255, 0], 15)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import os
os.listdir("test_images/")


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# Load the image
image = mpimg.imread('test_images/solidYellowCurve.jpg')

# First thing, let's gray out the image.
gray_img = grayscale(image)

# Now let's apply a Gaussian Blur to this image.
# I am using Kernel 5 because this image seems to be similar
# to the one from the previous quiz, where we settled on 5.
gaussian_img = gaussian_blur(gray_img, 5)

# Now let's apply Canny algorithm to detect the edges
canny_img = canny(gaussian_img, 50, 150)

# Now let's run the Hough lines
rho = 1 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # angular resolution in radians of the Hough grid
threshold = 20     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30 #minimum number of pixels making up a line
max_line_gap = 3    # maximum gap in pixels between connectable line segments

hough_img = hough_lines(canny_img, rho, theta, threshold, min_line_length, max_line_gap)

imshape = hough_img.shape
vertices = np.array([[(100,imshape[0]),(420, 330), (520, 330), (imshape[1]-60,imshape[0])]], dtype=np.int32)


reg_interest_img = region_of_interest(hough_img, vertices)

final_image = weighted_img(reg_interest_img, image)

# plt.imshow(gray_img, cmap='gray')
# plt.title("Grayscale")
# plt.show()

# plt.imshow(gaussian_img, cmap='gray')
# plt.title("Gaussian")
# plt.show()

# plt.imshow(canny_img, cmap='gray')
# plt.title("Canny")
# plt.show()

# plt.imshow(hough_img, cmap='gray')
# plt.title("Hough Lines")
# plt.show()

# plt.imshow(reg_interest_img, cmap='gray')
# plt.title("Hough lines with region of interes")
# plt.show()

# plt.imshow(final_image, cmap='gray')
# plt.title("Final Weighted Image")
# plt.show()

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    
    # First thing, let's gray out the image.
    gray_img = grayscale(image)

    # Now let's apply a Gaussian Blur to this image.
    # I am using Kernel 5 because this image seems to be similar
    # to the one from the previous quiz, where we settled on 5.
    gaussian_img = gaussian_blur(gray_img, 5)

    # Now let's apply Canny algorithm to detect the edges
    canny_img = canny(gaussian_img, 50, 150)

    # Now let's run the Hough lines
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 20     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30 #minimum number of pixels making up a line
    max_line_gap = 3    # maximum gap in pixels between connectable line segments

    hough_img = hough_lines(canny_img, rho, theta, threshold, min_line_length, max_line_gap)

    imshape = hough_img.shape
    vertices = np.array([[(100,imshape[0]),(420, 330), (520, 330), (imshape[1]-60,imshape[0])]], dtype=np.int32)

    reg_interest_img = region_of_interest(hough_img, vertices)

    final_image = weighted_img(reg_interest_img, image)

    return final_image
import time

