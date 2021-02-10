import numpy as np
from seguq.utils.segmodel_wrapper_keras import SegModelWrapper

class SegCommon:
    """
    Base class for segmentation.
    """
    def __init__(self, model_dir=None, model_prefix=None, model=None, enable_dropout=False):
        if model is not None:
            self.segmodel = model
        else:
            assert model_dir is not None and model_prefix is not None, 'model directory and prefix must be provided'
            self.model_dir = model_dir
            self.model_prefix = model_prefix
            self.segmodel = SegModelWrapper(model_directory=model_dir, prefix=model_prefix,
                                            enable_dropout=enable_dropout)

    def segment(self, images, batch_size=16):
        """
        Segments input images. Overwrite if required.
        """
        images = self.pre(images)
        predictive_prob = self._predict(images, batch_size)
        return self.post(predictive_prob)

    def pre(self, images):
        """
        Preprocessing of the images. You should overwrite if the model needs specific preprocessing.
        """
        input_shape = images.shape[1:]
        assert 16 * (input_shape[0] // 16) == input_shape[0], \
            'invalid dimension 0 of the bscan. Make sure it is divisibe by 16'
        assert 16 * (input_shape[1] // 16) == input_shape[1], \
            'invalid dimension 1 of the bscan. Make sure it is divisibe by 16'

        assert np.max(images) > 1, 'The pixel range should be [0, 255].'

        images = images/255.0
        images = images[:, :, :, np.newaxis]
        return images

    def post(self, predictive_prob):
        """
        Postprocessing of the prediction. Overwrite it if required.
        """
        return predictive_prob

    def _predict(self, images, batch_size):
        """
        Prediction. Overwrite it if required.
        """
        predictive_prob = self.segmodel.predict(images, batch_size)
        return predictive_prob


setattr(SegCommon, '__doc__', SegCommon.__doc__)
