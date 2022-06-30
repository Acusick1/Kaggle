from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi
from src.settings import DATA_PATH

api = KaggleApi()
api.authenticate()


def get_dataset(name, competition=True):

    dataset_path = DATA_PATH / name
    dataset_path.mkdir(exist_ok=True)

    if competition:
        api.competition_download_files(name, dataset_path)
    else:
        raise NotImplementedError("Not yet implemented dataset downloading for standalone datasets")

    # TODO: Is downloaded zip file always the name of the dataset?
    with ZipFile((dataset_path / name).with_suffix(".zip")) as zip_data:
        zip_data.extractall(dataset_path)

    return dataset_path


if __name__ == "__main__":

    dataset_name = "titanic"
    get_dataset(dataset_name)
