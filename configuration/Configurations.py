import json
import os
from .DataSourceConfigurations import DataSourceConfigurations

config_file = os.path.join(os.path.dirname(__file__), 'configurations.json')

class Configurations:
    def __init__(self):
        config = json.load(open(config_file))
        self.data_source_properties = DataSourceConfigurations(config['dataSource'])
        self.intent_home = config['intentHome']

