
import logging
import shutil
from setup import Settings

logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")

# function for deployment


def store_model_into_pickle(setup):
    """
    function for deployment

    copy the latest pickle file, 
    the latestscore.txt value, and 
    the ingestfiles.txt file into the 
    deployment directory


    Args:
        setup (_type_): _description_
    """
    file_move_list = [
        f"{setup.OUTPUT_FOLDER_PATH}/ingestedfiles.txt",
        f"{setup.OUTPUT_MODEL_PATH}/trainedmodel.joblib",
        f"{setup.OUTPUT_MODEL_PATH}/latestscore.txt"
    ]
    for file in file_move_list:
        shutil.copy(file,
                    setup.PROD_DEPLOYMENT_PATH)


if __name__ == '__main__':
    logging.info("model scoring")
    setup = Settings()
    store_model_into_pickle(setup)
