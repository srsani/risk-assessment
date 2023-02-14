"""
Author: Sohrab Sani
Date: February, 2021
This script used for plot and saveing confusionmatrix.png 
"""
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from diagnostics import model_predictions


from setup import Settings
setup = Settings()


def score_model():
    """
   Function for reporting

    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """

    df = pd.read_csv(f"{setup.OUTPUT_FOLDER_PATH}/finaldata.csv")
    X = df.drop(['corporation', 'exited'], axis=1)
    y = df['exited'].values

    y_pred, _ = model_predictions(X, y)

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    plot = disp.plot(cmap=plt.cm.Blues)
    plot.figure_.savefig(f"{setup.OUTPUT_MODEL_PATH}/confusionmatrix.png")


if __name__ == '__main__':
    score_model()
