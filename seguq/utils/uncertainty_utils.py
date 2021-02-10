import numpy as np
import cv2
import dataflow as df


def check_dropout(model):
    model_config = model.get_config()
    for i in range(len(model_config['layers'])):
        class_name = model_config['layers'][i]['class_name']
        if class_name == 'SpatialDropout2D' or class_name == 'Dropout':
            flags = model_config['layers'][i]['inbound_nodes'][0][0][-1]
            assert 'training' in flags and flags['training']==True, 'Dropout not enabled in based model'


def compute_entropy(predictive_prob):
    entropy_func = lambda x: -1 * np.sum(np.log(x + np.finfo(np.float32).eps) * x, axis=3)
    return entropy_func(predictive_prob)


def compute_num_classes(model_outputs):
    softmax_index = 0 if len(model_outputs) == 1 else 1
    return model_outputs[softmax_index].get_shape().as_list()[3]




class WarpAffineTransform(df.imgaug.Transform):
    def __init__(self, mat, dsize, interp=cv2.INTER_LINEAR,
                 borderMode=cv2.BORDER_CONSTANT, borderValue=0):
        super(WarpAffineTransform, self).__init__()
        self._init(locals())

    def apply_image(self, img):
        ret = cv2.warpAffine(img, self.mat, self.dsize,
                             flags=self.interp,
                             borderMode=self.borderMode,
                             borderValue=self.borderValue)
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def apply_coords(self, coords):
        coords = np.concatenate((coords, np.ones((coords.shape[0], 1), dtype='f4')), axis=1)
        coords = np.dot(coords, self.mat.T)
        return coords



class Rotation(df.imgaug.ImageAugmentor):
    """ Random rotate the image w.r.t a random center"""

    def __init__(self, deg, center,
                 interp=cv2.INTER_LINEAR,
                 border=cv2.BORDER_REPLICATE,  border_value=0):
        """
        Args:
            max_deg (float): max abs value of the rotation angle (in degree).
            center_range (tuple): (min, max) range of the random rotation center.
            interp: cv2 interpolation method
            border: cv2 border method
            step_deg (float): if not None, the stepping of the rotation
                angle. The rotation angle will be a multiple of step_deg. This
                option requires ``max_deg==180`` and step_deg has to be a divisor of 180)
            border_value: cv2 border value for border=cv2.BORDER_CONSTANT
        """
        #assert step_deg is None or (max_deg == 180 and max_deg % step_deg == 0)
        super(Rotation, self).__init__()
        self._init(locals())

    def get_transform(self, img):
        center = self.center# img.shape[1::-1] * self._rand_range(
            #self.center_range[0], self.center_range[1], (2,))
        deg= self.deg
        #deg = self._rand_range(-self.max_deg, self.max_deg)
        #if self.step_deg:
        #    deg = deg // self.step_deg * self.step_deg
        """
        The correct center is shape*0.5-0.5. This can be verified by:
        SHAPE = 7
        arr = np.random.rand(SHAPE, SHAPE)
        orig = arr
        c = SHAPE * 0.5 - 0.5
        c = (c, c)
        for k in range(4):
            mat = cv2.getRotationMatrix2D(c, 90, 1)
            arr = cv2.warpAffine(arr, mat, arr.shape)
        assert np.all(arr == orig)
        """
        mat = cv2.getRotationMatrix2D(tuple(center - 0.5), float(deg), 1)
        return WarpAffineTransform(
            mat, img.shape[1::-1], interp=self.interp,
            borderMode=self.border, borderValue=self.border_value)
