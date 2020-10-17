"""
DOCSTRING
"""
import cv2
import matplotlib
import numpy
import pandas

class Demo:
    """
    DOCSTRING
    """
    def circle_contour(self, image, contour):
        """
        DOCSTRING
        """
        image_with_ellipse = image.copy()
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2, cv2.CV_AA)
        return image_with_ellipse

    def find_biggest_contour(self, image):
        """
        DOCSTRING
        """
        image = image.copy()
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        mask = numpy.zeros(image.shape, numpy.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return biggest_contour, mask

    def find_strawberry(self, image):
        """
        DOCSTRING
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        max_dimension = max(image.shape)
        scale = 700 / max_dimension
        image = cv2.resize(image, None, fx=scale, fy=scale)
        image_blur = cv2.GaussianBlur(image, (7, 7), 0)
        image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
        min_red = numpy.array([0, 100, 80])
        max_red = numpy.array([10, 256, 256])
        mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)
        min_red2 = numpy.array([170, 100, 80])
        max_red2 = numpy.array([180, 256, 256])
        mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
        mask = mask1 + mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
        big_strawberry_contour, mask_strawberries = self.find_biggest_contour(mask_clean)
        overlay = self.overlay_mask(mask_clean, image)
        circled = self.circle_contour(overlay, big_strawberry_contour)
        self.show(circled)
        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        return bgr

    def overlay_mask(self, mask, image):
        """
        DOCSTRING
        """
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
        return img

    def show(self, image):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(10, 10))
        matplotlib.pyplot.imshow(image, interpolation='nearest')

class Working:
    """
    DOCSTRING
    """
    def __call__(self):
        image = cv2.imread('strawberries.jpg')
        image.shape
        m,n,r = image.shape
        arr = image.reshape(m*n, -1)
        df = pandas.DataFrame(arr, columns=['b', 'g', 'r'])
        df.describe()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, None, fx=1/3, fy=1/3)
        self.show(image)
        images = []
        for i in [0, 1, 2]:
            colour = image.copy()
            if i != 0: colour[:,:,0] = 0
            if i != 1: colour[:,:,1] = 0
            if i != 2: colour[:,:,2] = 0
            images.append(colour)
        self.show(numpy.vstack(images))
        self.show_rgb_hist(image)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        images = list()
        for i in [0, 1, 2]:
            colour = hsv.copy()
            if i != 0: colour[:,:,0] = 0
            if i != 1: colour[:,:,1] = 255
            if i != 2: colour[:,:,2] = 255
            images.append(colour)
        hsv_stack = numpy.vstack(images)
        rgb_stack = cv2.cvtColor(hsv_stack, cv2.COLOR_HSV2RGB)
        self.show(rgb_stack)
        matplotlib.rcParams.update({'font.size': 16})
        self.show_hsv_hist(hsv)
        image_blur = cv2.GaussianBlur(image, (7, 7), 0)
        self.show(image_blur)
        image_cropped = image[100:300, 200:500]
        self.show(image_cropped)
        image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
        min_red = numpy.array([0, 100, 80])
        max_red = numpy.array([10, 256, 256])
        image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
        min_red2 = numpy.array([170, 100, 80])
        max_red2 = numpy.array([180, 256, 256])
        image_red2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
        self.show_mask(image_red1)
        self.show_mask(image_red2)
        image_red = image_red1 + image_red2
        self.show_mask(image_red)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        image_red_closed = cv2.morphologyEx(image_red, cv2.MORPH_CLOSE, kernel)
        self.show_mask(image_red_closed)
        image_red_closed_then_opened = cv2.morphologyEx(image_red_closed, cv2.MORPH_OPEN, kernel)
        self.show_mask(image_red_closed_then_opened)
        big_contour, red_mask = self.find_biggest_contour(image_red_closed_then_opened)
        self.show_mask(red_mask)
        self.overlay_mask(red_mask, image)
        moments = cv2.moments(red_mask)
        centre_of_mass = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
        image_with_com = image.copy()
        cv2.circle(image_with_com, centre_of_mass, 10, (0, 255, 0), -1, cv2.CV_AA)
        self.show(image_with_com)
        image_with_ellipse = image.copy()
        ellipse = cv2.fitEllipse(big_contour)
        cv2.ellipse(image_with_ellipse, ellipse, (0, 255, 0), 2)
        self.show(image_with_ellipse)

    def find_biggest_contour(self, image):
        image = image.copy()
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1] 
        mask = numpy.zeros(image.shape, numpy.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return biggest_contour, mask
        
    def overlay_mask(self, mask, image):
        """
        DOCSTRING
        """
        rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
        self.show(img)

    def show(self, image):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(15, 15))
        matplotlib.pyplot.imshow(image, interpolation='nearest')
    
    def show_hsv(self, hsv):
        """
        DOCSTRING
        """
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.show(rgb)
    
    def show_hsv_hist(self, image):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(20, 3))
        histr = cv2.calcHist([image], [0], None, [180], [0, 180])
        matplotlib.pyplot.xlim([0, 180])
        colours = [matplotlib.colors.hsv_to_rgb((i/180, 1, 0.9)) for i in range(0, 180)]
        matplotlib.pyplot.bar(range(0, 180), histr, color=colours, edgecolor=colours, width=1)
        matplotlib.pyplot.title('Hue')
        matplotlib.pyplot.figure(figsize=(20, 3))
        histr = cv2.calcHist([image], [1], None, [256], [0, 256])
        matplotlib.pyplot.xlim([0, 256])
        colours = [matplotlib.colors.hsv_to_rgb((0, i/256, 1)) for i in range(0, 256)]
        matplotlib.pyplot.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
        matplotlib.pyplot.title('Saturation')
        matplotlib.pyplot.figure(figsize=(20, 3))
        histr = cv2.calcHist([image], [2], None, [256], [0, 256])
        matplotlib.pyplot.xlim([0, 256])
        colours = [matplotlib.colors.hsv_to_rgb((0, 1, i/256)) for i in range(0, 256)]
        matplotlib.pyplot.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
        matplotlib.pyplot.title('Value')

    def show_mask(self, mask):
        """
        DOCSTRING
        """
        matplotlib.pyplot.figure(figsize=(10, 10))
        matplotlib.pyplot.imshow(mask, cmap='gray')

    def show_rgb_hist(self, image):
        """
        DOCSTRING
        """
        colours = ('r','g','b')
        for i, c in enumerate(colours):
            matplotlib.pyplot.figure(figsize=(20, 4))
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            if c == 'r': colours = [((i/256, 0, 0)) for i in range(0, 256)]
            if c == 'g': colours = [((0, i/256, 0)) for i in range(0, 256)]
            if c == 'b': colours = [((0, 0, i/256)) for i in range(0, 256)]
            matplotlib.pyplot.bar(range(0, 256), histr, color=colours, edgecolor=colours, width=1)
            matplotlib.pyplot.show()

if __name__ == '__main__':
    image = cv2.imread('yo.jpg')
    result = Demo.find_strawberry(image)
    cv2.imwrite('yo2.jpg', result)
