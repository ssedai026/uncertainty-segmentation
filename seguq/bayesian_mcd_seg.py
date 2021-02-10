import numpy as np

from seguq.common_seg import SegCommon
from seguq.utils.uncertainty_utils import compute_entropy, check_dropout,compute_num_classes



class SegBayesianMCD(SegCommon):
    def __init__(self, T, model_dir, model_prefix, model=None):
        """
        Computes the uncertainty of segmentation based on monte carlo dropout.
        Works only if the base model has been trained with dropout.
        :param T: number of monte carlo samples.
        """
        self.T = T
        super(SegBayesianMCD, self).__init__(model_dir, model_prefix, model=model, enable_dropout=True)
        self.model = self.segmodel.get_model()
        check_dropout(self.model)

    def _predict(self, images, batch_size=16):
        """
        :param images:
        :param batch_size:
        :return:
        """
        N_class = compute_num_classes(self.model.outputs)
        predictive_prob_total = np.zeros((images.shape[0], images.shape[1], images.shape[2], N_class))
        for i in range(self.T):
            predictive_prob = self.model.predict(images, batch_size=batch_size)
            if (type(predictive_prob) is list):# some models may return logit, segmap
                predictive_prob = predictive_prob[1]

            predictive_prob_total += predictive_prob

        predictive_prob_average = predictive_prob_total / (self.T * 1.0)
        entropy = compute_entropy(predictive_prob_average)
        return predictive_prob_average, entropy

    def post(self, out):
        segmap, var = out
        return np.argmax(segmap, axis=3), var


setattr(SegBayesianMCD, '__doc__', SegBayesianMCD.__doc__)
