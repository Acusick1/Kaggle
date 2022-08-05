import pandas as pd
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from src.settings import DATA_PATH

API = KaggleApi()
API.authenticate()


def get_dataset(name, competition=True):

    dataset_path = DATA_PATH / name

    if not dataset_path.is_dir():

        dataset_path.mkdir()
        if competition:
            API.competition_download_files(name, dataset_path)
        else:
            raise NotImplementedError("Not yet implemented dataset downloading for standalone datasets")

    zip_file = (dataset_path / name).with_suffix(".zip")
    if zip_file.is_file():
        # TODO: Is downloaded zip file always the name of the dataset?
        with ZipFile(zip_file) as zip_data:
            zip_data.extractall(dataset_path)

        zip_file.unlink()

    return dataset_path


def load_dataset(name: str,
                 train_file: str = "train",
                 test_file: str = "test",
                 extension: str = ".csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw train and test datasets for a given dataset.
    :param name: dataset name
    :param train_file: name of training set file
    :param test_file: name of test set file
    :param extension: file format
    :return: train_dataset, test_dataset
    """
    path = get_dataset(name)
    raw_train_data = pd.read_csv((path / train_file).with_suffix(extension))
    raw_test_data = pd.read_csv((path / test_file).with_suffix(extension))

    return raw_train_data, raw_test_data


def main():

    dataset_name = "titanic"
    data_path = get_dataset(dataset_name)
    train, test = load_dataset(data_path)
    return train, test


if __name__ == "__main__":

    main()
