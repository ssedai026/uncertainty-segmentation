import os
import tensorflow as tf
from seguq.utils.custom_layers import Softmax4D
from tensorflow.keras.models import model_from_json


def enable_dropout(model, rate=None, custom_objects={}):
    """
    Enables the droput layer - used for monte carlo droput based uncertainty computation
    Note: the weights needs to be reloaded after calling this model

    >>> model = enable_dropout(model)
    >>> model.load_weights('path to model weight')
    :param model:
    :param rate:
    :param custom_objects:
    :return:
    """
    if(rate is not None): assert rate >= 0 and rate < 1, 'dropout rate is out of range'

    model_config = model.get_config()
    for i in range(len(model_config['layers'])):
        class_name = model_config['layers'][i]['class_name']
        if (class_name == 'SpatialDropout2D' or class_name =='Dropout' ):
            model_config['layers'][i]['inbound_nodes'][0][0][-1]['training'] = True
            if (rate is not None): model_config['layers'][i]['config']['rate'] = rate
            #print('dropout enabled')

    model = tf.keras.models.Model.from_config(model_config, custom_objects=custom_objects)
    return model


class SegModelWrapper(object):

    def __init__(self, prefix, model_directory, model=None, custom_objects=None, **kwarg):

        if(custom_objects is None):
            custom_objects = {"Softmax4D": Softmax4D}
        else:
            custom_objects["Softmax4D"] = Softmax4D


        self.model = self.load_model(prefix, model_directory, model=model, custom_objects=custom_objects, **kwarg)

    def predict(self, images, batch_size=10):
        out = self.model.predict(images, batch_size)
        return out

    def predict_on_batch(self, images):
        return self.model.predict_on_batch(images)

    def get_model(self):
        return self.model

    @staticmethod
    def load_model(file_name_prefix, model_directory, model, custom_objects=None, **kwarg):
        """
        Loads keras model saved as hd5 and json defination
        :param file_name_prefix:  prefix of the file name
        :return:
        """

        print(file_name_prefix)
        print(model_directory)

        json = os.path.join(os.getcwd(), model_directory, file_name_prefix + '.json')
        with open(json) as j:
            json_string = j.read()

        if (model is None):
            amodel = model_from_json(json_string, custom_objects=custom_objects)
        else:
            amodel = model

        if ('enable_dropout' in kwarg and kwarg['enable_dropout'] == True):
            rate = kwarg['dropout_rate'] if 'dropout_rate' in kwarg else None
            print('Loading model by enabling dropout.')
            amodel = enable_dropout(amodel, custom_objects=custom_objects, rate=rate)

        amodel.load_weights(os.path.join(model_directory, file_name_prefix + '.hd5'))
        return amodel



