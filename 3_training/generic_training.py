import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras_nlp.layers import TransformerEncoder, SinePositionEncoding, FNetEncoder
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

import sys

# sys.path.append('./architectures')


def load_data(arquivo_metadados, pasta_base, split):
    metadados = pd.read_csv(arquivo_metadados)
    # Filtrar pelo split desejado (mdpi_level_5 contém 'train', 'val', 'test')
    metadados = metadados[(metadados['mdpi_level_5'] == split) 
                          & (metadados['mdpi_level_2'] == "not_inverted")
                          & (metadados['mdpi_level_3'] == "world")
                          & (metadados['mdpi_level_4'] == "lite")]
    dados = []
    rotulos = []
    for _, linha in metadados.iterrows():
        id_keypoint = linha['id_keypoint']
        
        # Determinar o rótulo baseado na coluna mdpi_level_1
        # 'quedas' = 1, 'not_quedas' = 0 (ou outra categoria)
        if linha['mdpi_level_1'] == 'quedas':
            rotulo = 1  # queda
        else:
            rotulo = 0  # não queda

        if not str(id_keypoint).endswith('.csv'):
            caminho_arquivo = os.path.join(pasta_base, f"{id_keypoint}.csv")
        else:
            caminho_arquivo = os.path.join(pasta_base, f"{id_keypoint}")
        
        if os.path.exists(caminho_arquivo):
            try:
                df_arquivo = pd.read_csv(caminho_arquivo)
                array_arquivo = df_arquivo.to_numpy()                
                if array_arquivo.shape == (44, 132):
                    dados.append(array_arquivo)
                    rotulos.append(rotulo)
                else:
                    print(f"Aviso: Arquivo {id_keypoint} tem dimensões {array_arquivo.shape}, esperado (44, 132)")
                    
            except Exception as e:
                print(f"Erro ao ler arquivo {caminho_arquivo}: {e}")
        else:
            print(f"Arquivo não encontrado: {caminho_arquivo}")
    
    return np.array(dados), np.array(rotulos)





def create_model():
    """
    Cria e compila o modelo de rede neural
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
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model





def plot_training_history(history):
    """
    Plota gráficos de accuracy e loss do treinamento
    """
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_dados, test_rotulos):
    """
    Avalia o modelo no conjunto de teste e gera métricas
    """
    # Predições no conjunto de teste
    y_pred = model.predict(test_dados)

    # Converter probabilidades para previsões binárias
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Relatório de classificação
    print("Classification Report:")
    print(classification_report(test_rotulos, y_pred_binary))

    # Matriz de confusão
    conf_matrix = confusion_matrix(test_rotulos, y_pred_binary)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot da curva ROC
    fpr, tpr, _ = roc_curve(test_rotulos, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return y_pred, y_pred_binary, conf_matrix, roc_auc


def print_data_distribution(train_rotulos, val_rotulos, test_rotulos):
    """
    Imprime a distribuição dos dados entre classes
    """
    print(f"Distribuição treinamento - Quedas: {np.sum(train_rotulos == 1)}, Não-quedas: {np.sum(train_rotulos == 0)}")
    print(f"Distribuição validação - Quedas: {np.sum(val_rotulos == 1)}, Não-quedas: {np.sum(val_rotulos == 0)}")
    print(f"Distribuição teste - Quedas: {np.sum(test_rotulos == 1)}, Não-quedas: {np.sum(test_rotulos == 0)}")


def main():
    # Configurações
    arquivo_metadados = "../metadata/2_keypoints.csv" 
    pasta_base = "../data/2_keypoints"
    
    # Carregar dados
    print("Carregando dados de treinamento...")
    train_dados, train_rotulos = load_data(arquivo_metadados, pasta_base, 'train')
    print("Carregando dados de validação...")
    val_dados, val_rotulos = load_data(arquivo_metadados, pasta_base, 'val')
    print("Carregando dados de teste...")
    test_dados, test_rotulos = load_data(arquivo_metadados, pasta_base, 'test')


    print(f"Treinamento: {len(train_dados)} amostras")
    print(f"Validação: {len(val_dados)} amostras")
    print(f"Teste: {len(test_dados)} amostras")
    print_data_distribution(train_rotulos, val_rotulos, test_rotulos)


    # Criar modelo
    model = create_model()
    
    print('///////////////////////////')
    model.summary()

    # Configurar early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=30,  
        restore_best_weights=True
    )

    # Treinar modelo
    print("Iniciando treinamento...")
    history = model.fit(
        train_dados, train_rotulos, 
        epochs=1000, 
        batch_size=32, 
        validation_data=(val_dados, val_rotulos), 
        callbacks=[early_stopping]
    )

    # Plotar histórico de treinamento
    plot_training_history(history)

    # Avaliar modelo
    evaluate_model(model, test_dados, test_rotulos)

    # Estatísticas do modelo
    num_params = model.count_params()
    print(f'Number of parameters in the saved model: {num_params}')

    # Salvar modelo
    model.save('../data/3_models/trained_model.keras')
    print("Modelo salvo com sucesso!")


if __name__ == '__main__':
    main()