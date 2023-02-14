
import logging
import os
import pandas as pd
import json
from datetime import datetime


from setup import Settings
setup = Settings()


logging.basicConfig(
    filename='./logs/results.log',
    level=logging.INFO,
    filemode='a+',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def merge_multiple_dataframe(setup):
    """
    merge_multiple_dataframe _summary_

    _extended_summary_

    Args:
        setup (_type_): _description_
    """

    df = pd.DataFrame()
    df_dict = {}

    logging.info(f"Reading files from {setup.INPUT_FOLDER_PATH}")
    for file in os.listdir(setup.INPUT_FOLDER_PATH):
        file_path = f"{setup.INPUT_FOLDER_PATH}/{file}"
        df_tmp = pd.read_csv(file_path)
        df = pd.concat([df, df_tmp], ignore_index=True)
        df_dict[file] = df_tmp.shape

    logging.info("dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=1)

    logging.info("saving ingested metadata")
    with open(f"{setup.OUTPUT_FOLDER_PATH}/ingestedfiles.txt", "w") as f:
        # f.write(
        #     f"\ningestion date: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        # f.write(f"\nfinal df shape: {df.shape}\n")
        f.write(json.dumps(df_dict, indent=4))

    logging.info("saving ingested data")
    df.to_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv", index=False)


if __name__ == '__main__':
    setup = Settings()
    merge_multiple_dataframe(setup)
