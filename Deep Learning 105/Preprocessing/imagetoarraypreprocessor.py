from tensorflow.keras.preprocessing.image import img_to_array

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        # apply keras utility function that correctly rearranges dimensions of image
        return img_to_array(image, data_format=self.dataFormat)