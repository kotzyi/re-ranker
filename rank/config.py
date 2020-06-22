from yaml2object import YAMLObject


class ModelConfig(metaclass=YAMLObject):
    source = "../config/model.yml"
    namespace = 'model'
