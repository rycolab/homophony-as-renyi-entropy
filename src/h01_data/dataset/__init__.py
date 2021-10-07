from .celex import Celex


def get_dataset_cls(dataset_name):
    datasets = {
        'celex': Celex,
    }
    return datasets[dataset_name]


def get_languages(dataset_name, data_path):
    dataset_cls = get_dataset_cls(dataset_name)
    return dataset_cls.get_languages(data_path)
