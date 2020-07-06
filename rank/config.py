from yaml2object import YAMLObject


class DDPGConfig(metaclass=YAMLObject):
    source = "../config/model.yml"
    namespace = 'ddpg'


class TD3Config(metaclass=YAMLObject):
    source = "../config/model.yml"
    namespace = 'td3'
