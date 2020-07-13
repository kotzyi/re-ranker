from yaml2object import YAMLObject


class DDPGConfig(metaclass=YAMLObject):
    source = "../config/model.yml"
    namespace = 'ddpg'


class TD3Config(metaclass=YAMLObject):
    source = "../config/model.yml"
    namespace = 'td3'


class CardTypeV2Config(metaclass=YAMLObject):
    source = "../config/env.yml"
    namespace = 'card_v2'


class CardTypeV3Config(metaclass=YAMLObject):
    source = "../config/env.yml"
    namespace = 'card_v3'
