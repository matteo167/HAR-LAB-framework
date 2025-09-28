import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import FNetEncoder, SinePositionEncoding

def create_model():
    """
    Cria modelo com arquitetura FNet
    """
    input = keras.Input(shape=(44, 132))
    position_embeddings = SinePositionEncoding()(input)
    input_position = input + position_embeddings
    x = FNetEncoder(intermediate_dim=64)(input_position)
    x = layers.Permute((2, 1))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(22, activation="relu")(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=input, outputs=output)
    return model

def get_model_info():
    """Retorna informações sobre o modelo"""
    return {
        'name': 'FNet',
        'description': 'Modelo FNet com positional encoding',
        'parameters': '~50K params'
    }