#EN
The script modulos_anomalias_full.py is a module for anomaly detection in well sensor data. It leverages machine learning techniques, specifically LSTM (Long Short-Term Memory) neural networks, to identify and classify anomalies in the data. Here's a detailed breakdown of what the code does:
Data Initialization and Preparation:

    The code starts by importing necessary libraries, such as copy, gc, logging, os, json, pandas, numpy, MinMaxScaler from sklearn, and several functions from tensorflow and keras.
    It checks if the message (msg) contains a payload and initializes variables and lists to store sensor data.
    A list of variables of interest (varnamelist) is defined, and data structures are initialized to store historical and current sensor data.

Data Processing:

    The code inspects the state of the received data (normal, analyzing, or invalid) to determine how to process it.
    If the data is normal and there are fewer than 500 samples, the data is accumulated. If there are already more than 500 samples, a FIFO (First In, First Out) policy is applied to maintain a constant data window size.

Model Training and Evaluation:

    If a trained network exists for the current valve state, the code loads the saved model weights and scaler, and tests the current data.
    The model is an LSTM neural network trained to predict sensor values. The Mean Absolute Error (MAE) is used to evaluate the model's performance.
    If the average loss is within an acceptable threshold, the data is considered normal. Otherwise, it is classified as anomalous with varying risk levels (low, medium, high).

Model Update:

    If 500 new data points are inserted, the code retrains the LSTM neural network using the accumulated data.
    The trained model's weights, scaler, and tolerance are saved in the message payload for future use.

Cleanup and Finalization:

    After processing, computed data is cleared, and the message is updated with the new state of the model and data.


#PTBR 
O arquivo modulos_anomalias_full.py é um módulo de detecção de anomalias em dados de sensores de poços. Ele utiliza técnicas de aprendizado de máquina, especificamente redes neurais LSTM (Long Short-Term Memory), para identificar e classificar anomalias nos dados. Aqui está uma descrição detalhada do que o código faz:

    Inicialização e Preparação dos Dados:
        O código começa importando bibliotecas necessárias, como copy, gc, logging, os, json, pandas, numpy, MinMaxScaler da sklearn, e várias funções do tensorflow e keras.
        Ele verifica se a mensagem (msg) contém um payload e inicializa variáveis e listas para armazenar os dados dos sensores.
        Define uma lista de variáveis de interesse (varnamelist) e inicializa a estrutura de dados para armazenar os dados históricos e atuais dos sensores.

    Processamento dos Dados:
        O código verifica o estado dos dados recebidos (normal, analisando, dados inválidos) e decide como processá-los.
        Se os dados são normais e há menos de 500 amostras, eles são acumulados. Se já existem mais de 500 amostras, o código aplica uma política FIFO (First In, First Out) para manter o tamanho da janela de dados constante.

    Treinamento e Avaliação do Modelo:
        Se há uma rede treinada para o estado atual das válvulas, o código carrega os pesos e o scaler do modelo salvo e testa os dados atuais.
        O modelo é uma rede neural LSTM que é treinada para prever os valores dos sensores. A perda média absoluta (MAE) é usada para avaliar a performance do modelo.
        Se a perda média está dentro de um limite aceitável, os dados são considerados normais. Caso contrário, são classificados como anômalos com diferentes níveis de risco (baixo, médio, alto).

    Atualização do Modelo:
        Se 500 novos dados foram inseridos, o código treina novamente a rede neural com os dados acumulados.
        Os pesos do modelo treinado, o scaler e a tolerância são salvos no payload da mensagem para uso futuro.

    Limpeza e Finalização:
        Após o processamento, os dados já computados são apagados e a mensagem é atualizada com o novo estado do modelo e dos dados.
