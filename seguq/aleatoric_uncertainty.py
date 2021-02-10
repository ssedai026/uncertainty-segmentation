import dataflow as df
import numpy as np
from dataflow.dataflow.imgaug.imgproc import Contrast, Brightness, GaussianBlur
from seguq.common_seg import SegCommon
from seguq.utils.uncertainty_utils import compute_entropy, compute_num_classes, Rotation


class AleatoricUncertainty(SegCommon):
    def __init__(self, model_dir, model_prefix, model=None, augmentors_factory=None):
        """
        Computes aleatoric uncertainty using test time augmentation.
        :param augmentors_func:  a function call that should return a list of augmentors that follows dataflow API.
        """
        super(AleatoricUncertainty, self).__init__(model_dir, model_prefix, model=model)
        # if augmentors is None:
        #    self.augmenters = [(Contrast((0, 3)), None), (GaussianBlur(), None), (Brightness(20), None)]
        # else:

        self.augmentors_factory = augmentors_factory
        self.model = self.segmodel.get_model()
        self.max_deg = 5
        self.center_range = 0, 1

    def get_augmentors(self, image):
        return [(Contrast((0, 3)), None), (GaussianBlur(), None), (Brightness(10), None), self.get_rand_rotation(image)]

    def get_rand_rotation(self, image):
        deg = self._rand_range(-self.max_deg, self.max_deg)
        center = image.shape[1::-1] * self._rand_range(
            self.center_range[0], self.center_range[1], (2,))

        r = Rotation(deg, center)
        rinv = Rotation(-deg, center)
        return r, rinv

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Generate uniform float random number between low and high using `self.rng`.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        rng = df.utils.get_rng()

        return rng.uniform(low, high, size).astype("float32")

    def _predict(self, images, batch_size=1):
        """
        :param images:
        :return:
        """
        N_class = compute_num_classes(self.model.outputs)
        predictive_prob_average_total = np.zeros((images.shape[0], images.shape[1], images.shape[2], N_class))
        entropy_total = np.zeros((images.shape[0], images.shape[1], images.shape[2]))
        for i in range(len(images)):
            predictive_prob_total = np.zeros((1, images.shape[1], images.shape[2], N_class))
            augmentors = self.get_augmentors(
                image=images[i, :, :, :]) if self.augmentors_factory is None else self.augmentors_factory()
            for augmentor in augmentors:
                transformer, inverse_transformer = augmentor
                image_transformed = transformer.augment(images[i, :, :, :] * 255.)
                image_transformed = image_transformed[np.newaxis, :, :, :] / 255.
                predictive_prob = self.model.predict(image_transformed, batch_size=1)
                if inverse_transformer is not None:
                    predictive_prob = inverse_transformer.augment(predictive_prob[0])
                    predictive_prob = predictive_prob[np.newaxis, :, :, :]
                predictive_prob_total += predictive_prob

            predictive_prob_average = predictive_prob_total / (len(augmentors) * 1.0)
            predictive_prob_average_total[i, :, :, :] = predictive_prob_average
            entropy_total[i, :, :] = compute_entropy(predictive_prob_average)

        return predictive_prob_average_total, entropy_total

    def post(self, out):
        segmap, var = out
        return np.argmax(segmap, axis=3), var


setattr(AleatoricUncertainty, '__doc__', AleatoricUncertainty.__doc__)
