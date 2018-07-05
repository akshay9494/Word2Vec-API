from .FileConfigurations import FileConfigurations

class DataSourceConfigurations:
    def __init__(self, data_source_properties):
        # self.data_source_properties = data_source_properties
        if 'file' in data_source_properties.keys():
            self.file_properties = FileConfigurations(data_source_properties['file'])
            self.data_source = 'file'
        if 'existingEmbedding' in data_source_properties.keys():
            self.existing_embedding_path = data_source_properties['existingEmbedding']