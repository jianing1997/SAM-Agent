import logging
import os
import pickle
from pathlib import Path
from typing import List
from typing import Tuple

import numpy as np
import tensorflow as tf

from . import config

MODEL_BASE_DIR = Path(__file__).resolve().parent
model_paths = {
    "DECIMER": str(MODEL_BASE_DIR / "DECIMER-V2" / "DECIMER_model"),
    "DECIMER_HandDrawn": str(MODEL_BASE_DIR / "DECIMER-V2" / "DECIMER_HandDrawn_model"),
}
def get_models(model_paths: dict):
    """Download and load models from the provided URLs.

    This function downloads models from the provided URLs to a default location,
    then loads tokenizers and TensorFlow saved models.

    Args:
        model_urls (dict): A dictionary containing model names as keys and their corresponding URLs as values.

    Returns:
        tuple: A tuple containing loaded tokenizer and TensorFlow saved models.
            - tokenizer (object): Tokenizer for DECIMER model.
            - DECIMER_V2 (tf.saved_model): TensorFlow saved model for DECIMER.
            - DECIMER_Hand_drawn (tf.saved_model): TensorFlow saved model for DECIMER HandDrawn.
    """

    # Load tokenizers
    tokenizer_path = os.path.join(
        model_paths["DECIMER"], "assets", "tokenizer_SMILES.pkl"
    )
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    # Load DECIMER models
    DECIMER_V2 = tf.saved_model.load(model_paths["DECIMER"])
    DECIMER_Hand_drawn = tf.saved_model.load(model_paths["DECIMER_HandDrawn"])

    return tokenizer, DECIMER_V2, DECIMER_Hand_drawn

tokenizer, DECIMER_V2, DECIMER_Hand_drawn = get_models(model_paths)

def predict_SMILES(
        image_input: [str, np.ndarray], confidence: bool = False, hand_drawn: bool = False
) -> str:
    """Predicts SMILES representation of a molecule depicted in the given image.

    Args:
        image_input (str or np.ndarray): Path of chemical structure depiction image or a numpy array representing the image.
        confidence (bool): Flag to indicate whether to return confidence values along with SMILES prediction.
        hand_drawn (bool): Flag to indicate whether the molecule in the image is hand-drawn.

    Returns:
        str: SMILES representation of the molecule in the input image, optionally with confidence values.
    """
    chemical_structure = config.decode_image(image_input)

    model = DECIMER_Hand_drawn if hand_drawn else DECIMER_V2
    predicted_tokens, confidence_values = model(tf.constant(chemical_structure))

    predicted_SMILES = decoder(detokenize_output(predicted_tokens))

    if confidence:
        predicted_SMILES_with_confidence = detokenize_output_add_confidence(
            predicted_tokens, confidence_values
        )
        return predicted_SMILES, predicted_SMILES_with_confidence

    return predicted_SMILES

def detokenize_output(predicted_array: int) -> str:
    """This function takes the predicted tokens from the DECIMER model and
    returns the decoded SMILES string.

    Args:
        predicted_array (int): Predicted tokens from DECIMER

    Returns:
        (str): SMILES representation of the molecule
    """
    outputs = [tokenizer.index_word[i] for i in predicted_array[0].numpy()]
    prediction = (
        "".join([str(elem) for elem in outputs])
        .replace("<start>", "")
        .replace("<end>", "")
    )
    return prediction

def detokenize_output_add_confidence(
    predicted_array: tf.Tensor,
    confidence_array: tf.Tensor,
) -> List[Tuple[str, float]]:
    """This function takes the predicted array of tokens as well as the
    confidence values returned by the Transformer Decoder and returns a list of
    tuples that contain each token of the predicted SMILES string and the
    confidence value.

    Args:
        predicted_array (tf.Tensor): Transformer Decoder output array (predicted tokens)

    Returns:
        str: SMILES string
    """
    prediction_with_confidence = [
        (
            tokenizer.index_word[predicted_array[0].numpy()[i]],
            confidence_array[i].numpy(),
        )
        for i in range(len(confidence_array))
    ]
    # remove start and end tokens
    prediction_with_confidence_ = prediction_with_confidence[1:-1]

    decoded_prediction_with_confidence = list(
        [(decoder(tok), conf) for tok, conf in prediction_with_confidence_]
    )

    return decoded_prediction_with_confidence

def decoder(predictions):
    modified = (
        predictions.replace("!", "1")
        .replace("$", "2")
        .replace("^", "3")
        .replace("<", "4")
        .replace(">", "5")
        .replace("?", "6")
        .replace("£", "7")
        .replace("¢", "8")
        .replace("€", "9")
        .replace("§", "0")
    )
    return modified

