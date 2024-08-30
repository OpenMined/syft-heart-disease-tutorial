"""Collection of utility functions used throughout the ML experiments
to reduce boilerplate in the notebooks."""

import syft as sy
import joblib
import matplotlib.pyplot as plt

from IPython.display import display
from io import BytesIO
from pathlib import Path
from typing import Union
from sklearn.base import BaseEstimator

from itertools import product
from sklearn.metrics import ConfusionMatrixDisplay


def check_status_last_code_requests(datasites: dict[str, sy.DatasiteClient]) -> None:
    """display status message of last code request sent to each datasite"""
    for name, datasite in datasites.items():
        print(f"Datasite: {name}")
        display(datasite.code[-1].status.get_status_message())  # type: ignore


def get_model_file(datasite_name: str) -> str:
    return f"{datasite_name.replace(".", "").replace(" ", "_").lower()}_model.jbl"


def dump_model(
    datasite_name: str, model_buffer: BytesIO, root: str = "./models"
) -> str:
    """Store a serialised ml model in a BytesIO buffer into a binary file on disk"""
    model_folder = Path(root)
    model_folder.mkdir(exist_ok=True)

    filename = get_model_file(datasite_name=datasite_name)
    filepath = model_folder / filename
    with open(filepath, "wb") as model_file:
        model_file.write(model_buffer.getbuffer())

    return f"Model saved in {filepath}"


def load_model(model_filepath: Union[str, Path]) -> BaseEstimator:
    with open(model_filepath, "rb") as model_file:
        return joblib.load(model_file)


def load_model_from_buffer(model_buffer: BytesIO):
    """Load model into memory from Joblib BytesIO buffer"""
    model_buffer.seek(0)
    return joblib.load(model_buffer)


def load_models(
    datasites: dict[str, sy.DatasiteClient], root: str = "./models"
) -> dict[str, BaseEstimator]:
    model_folder = Path(root)
    models = {}
    for name in datasites:
        filename = get_model_file(datasite_name=name)
        filepath = model_folder / filename
        models[name] = load_model(filepath)
    return models


def serialize_and_upload(
    model: BaseEstimator, to: sy.DatasiteClient
) -> sy.ActionObject:
    """Serialise and upload an ML model to a target the datasite"""

    model_buffer = BytesIO()
    joblib.dump(model, model_buffer)
    model_buffer.seek(0)
    model_action_object = sy.ActionObject.from_obj(model_buffer)
    return model_action_object.send(to)


def plot_all_confusion_matrices(cms):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")

    for coords, (name, cf_matrix) in zip(product(range(2), repeat=2), cms.items()):
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["Absence", "Presence"])
        disp.plot(ax=axes[*coords], xticks_rotation=45)
        disp.ax_.set_title(name)
        disp.im_.colorbar.remove()  # type: ignore
        disp.ax_.set_xlabel("")
        if coords[1] != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.4, -0.05, "Predicted label", ha="left")
    plt.tight_layout()

    fig.colorbar(disp.im_, ax=axes)
    return fig
