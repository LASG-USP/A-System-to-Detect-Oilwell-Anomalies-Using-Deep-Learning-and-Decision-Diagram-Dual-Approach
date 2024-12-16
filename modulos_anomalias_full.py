import copy

# temporario, apagar depois
if(('payload' not in msg) or (msg['payload'] is None)):
    msg['payload'] = {}
if('temp' not in msg['payload']):
    msg['payload']['temp']= {'treinos':0,'anoms':0,'auxnum':0}

# presets
numDataJump = 5
flag_IAmodel = 0
well_name = list(msg['data'].keys())[0]
varnamelist = ['estado', 'PPDG', 'TPDG', 'PTPT', 'TTPT', 'PPT']
#varnamelist = []
for chave_sensor in list(msg['data'][well_name]['IA'].keys()):
    if (len(msg['data'][well_name]['IA'][chave_sensor]) > 5):
        break #varnamelist.append(chave_sensor)

# Define o nome da rede segundo a disposição de válvulas
nomerede = 'rede'+msg['data'][well_name]['cval']
nome_IA = well_name

# Inicializando a variavel dados_anterior
# dados_anterior vai receber os dados normais ja salvos no Redis (em msg['IA']) de bateladas anteriores (caso exista)
# e completar com os valores recebidos na ultima batelada, presentes em msg['data'][well_name]['IA'] (restante do codigo)
if( ('PPDG' not in msg['IA']) or (len(msg['IA']['PPDG']) <= 0) ):
    msg['IA'] = { 'count': 0 }
    for varname in varnamelist:
        msg['IA'][varname] = []

msg['IA']['cval'] = [msg['data'][well_name]['IA']['valname']]
msg['IA']['AI_state'] = 'Normal'

# Neste loop completa-se com os valores recebidos na ultima batelada, presentes em msg['data'][well_name]['IA']
# verifica-se se ocorreu mudanças de valvulas (modo analyzing) ou se ha dados anomalos (nao sao inseridos para treinamento)
for j in range(len(msg['data'][well_name]['IA'][chave_sensor])):
    
    # caso o poco esteja em analyzing (mudança de estado de válvulas), zere dados_anterior
    if ( msg['data'][well_name]['IA']['estado'][j].lower() == 'analyzing' ):
        #continue          #################<<<<<<<<<<<<#<#<#<#<#<#<#<#<<<<<<<<<<<<<<<<<<##########
        msg['IA']['count'] = 0
        for varname in varnamelist:
            msg['IA'][varname] = []

    # caso o poco nao esteja normal e nem em analyzing (ou seja, esteja anomalo)
    #elif ( msg['data'][well_name]['IA']['estado'][j] !== 'Normal' ):
    #    continue
    
    elif ( msg['data'][well_name]['IA']['estado'][j] == 'Invalid data' ):
        msg['IA']['AI_state'] = 'Invalid data'
        msg['IA']['AI_tolerance'] = 0
        msg['IA']['AI_error'] = 0
        continue
    
    # caso os dados sejam normais mas nao ha '500' dados, acumule
    elif ((msg['IA']['count']<=500) or (len(msg['IA']['estado'])<=500)):
        if((msg['data'][well_name]['IA']['estado'][j].lower() == 'normal' )):
            msg['IA']['count'] = msg['IA']['count']+1
            for varname in varnamelist:
                msg['IA'][varname].append(msg['data'][well_name]['IA'][varname][j])

    # caso os dados sejam normais e ja existem mais de 500 dados
    # adicione como ultimo e remova o primeiro (LIFO last in last out)
    else:
        msg['IA']['count'] = msg['IA']['count']+1
        for varname in varnamelist:
            msg['IA'][varname].append(msg['data'][well_name]['IA'][varname][j])
            msg['IA'][varname].pop(0)

    if((msg['IA']['count']>=500) and (len(msg['IA']['estado'])>=500)):
        #PREPARATIVO IA

        # agora que o dado foi salvo, verifica-se a existencia de uma rede treinada para o mesmo estado de valvula
        if( ('payload' not in msg) or (msg['payload'] is None) or (nomerede not in msg['payload']) or  len(msg['payload'][nomerede]) == 0):
            rede = 2 #nao ha rede treinada para este estado de valvula (nao existe msg['payload'][nomerede])
        # Caso seja a primeira iteração
        elif(msg['IA']['count']<100):
            rede = 0 #nao ha dados suficientes
        else:
            rede = 1 #existe uma rede treinada para este estado de valvula
        
        msg['IA']['rede'] = rede
        
        if(flag_IAmodel==0):
            flag_IAmodel = 1
            # Preparativo IA
            import gc
            import logging
            logging.getLogger('tensorflow').disabled = True
            import os
            from json import JSONEncoder
            import pandas as pd
            import numpy as np
            from sklearn.preprocessing import MinMaxScaler
            import codecs, json 
            from numpy.random import seed
            import tensorflow as tf
            tf.keras.backend.clear_session()
            try:
                tf.logging.set_verbosity(tf.logging.ERROR)
                tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
            except:
                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
            from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
            from keras.models import Model
            from keras import regularizers, callbacks
            
            # scaler para normalizacao
            scaler = MinMaxScaler()
            
            # define the autoencoder network model
            shape1 = 1
            shape2 = 5
            inputs = Input(shape=(shape1, shape2))
            L1 = LSTM(16, activation='relu', return_sequences=True,kernel_regularizer=regularizers.l2(0.00))(inputs)
            L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
            L3 = RepeatVector(shape1)(L2)
            L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
            L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
            output = TimeDistributed(Dense(shape2))(L5)
            model = Model(inputs=inputs, outputs=output)

            # create the autoencoder model
            model.compile(optimizer='adam', loss='mae')
            
            # settings to fit the model to the data
            nb_epochs = 200
            batch_size = 150
            
            #teste das versões
            #import sklearn
            import sys
            #print(sklearn.__version__) #sklearn
            #print(tf.__version__) #tensorflow
            print("User Current Version:-", sys.version) #python
            
        # caso exista uma rede treinada, teste a cada 5 (=numDataJump) dados
        if( (msg['IA']['count']%numDataJump == 0) and (msg['IA']['rede']==1) ):
            # teste da IA!
            li = [msg['IA']['PPDG'],msg['IA']['TPDG'],msg['IA']['PTPT'],msg['IA']['TTPT'],msg['IA']['PPT']]
            li = np.asarray(li)
            li = li.transpose()
            train = pd.DataFrame(data = li, columns= ["PPDG", "TPDG", "PTPT", "TTPT", "PPT"])
            for key in train.keys():
                train[key] = train[key].fillna(np.nanmean(np.array(train[key])))
            
            train = train.reset_index(drop=True)
        
            test_scaler = json.loads(msg['payload'][nomerede]['scaler'])
    
            for scalerkeys in test_scaler[0].keys():
                setattr(scaler,scalerkeys,test_scaler[0][scalerkeys])
            
            try:
                X_train = scaler.transform(train)
            except:
                # rede nao treinada, pois nao foi possivel recria-la
                msg['payload'].pop(nomerede, None)
                msg['IA']['rede']=0
            else:
                # reshape inputs for LSTM [samples, timesteps, features]
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                # define the autoencoder network model
               
                # create the autoencoder model
                ark = json.loads(msg['payload'][nomerede]['weights'])
                ark = np.array(ark,dtype=object)
                for i in range(len(ark)):
                    ark[i] = np.array(ark[i])
                
                try:
                    model.set_weights(ark)
                    # plot the loss distribution of the training set
                    X_pred = model.predict(X_train[(len(X_train)-3):(len(X_train)-1)])
                except:
                    # rede nao treinada, pois nao foi possivel recria-la
                    msg['payload'].pop(nomerede, None)
                    msg['IA']['rede']=0
                else:
                    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                    X_pred = pd.DataFrame(X_pred, columns=train.columns)
                    X_pred.index = train[0:len(X_pred)].index
                    
                    scored_train = pd.DataFrame(index=train.index)
                    Xtrain_t = X_train[(len(X_train)-3):(len(X_train)-1)]
                    Xtrain_t = Xtrain_t.reshape(Xtrain_t.shape[0], Xtrain_t.shape[2])
                    scored_train['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain_t), axis = 1)
                    
                    error_distance = (scored_train['Loss_mae'].mean() / msg['payload'][nomerede]['tolerance'])
                    
                    if(error_distance <= 1):
                    	cur_AIstate = 'Normal'
                    elif(( error_distance > 1)and( error_distance < 1.3 )):
                    	cur_AIstate = 'Anomalous-low risk'
                    elif(( error_distance >= 1.3 ) and ( error_distance < 1.7 )):
                    	cur_AIstate= 'Anomalous-med risk'
                    elif( error_distance >= 1.7 ):
                    	cur_AIstate = 'Anomalous-high risk'
                    
                    if(('log_teste' not in msg['payload']) or (msg['payload']['log_teste'] is None) or  len(msg['payload']['log_teste']) == 0):
                        msg['payload']['log_teste'] = []
                    msg['payload']['log_teste'].append(( j + 1 + msg['payload']['temp']['auxnum'] , cur_AIstate ))
                    
                    if((msg['IA']['AI_state']=='Normal') and (( error_distance > 1)and( error_distance < 1.3 ))):
                    	msg['IA']['AI_state'] = 'Anomalous-low risk'
                    elif((msg['IA']['AI_state']=='Normal') and (( error_distance >= 1.3 ) and ( error_distance < 1.7 ))):
                    	msg['IA']['AI_state'] = 'Anomalous-med risk'
                    elif( error_distance >= 1.7 ):
                    	msg['IA']['AI_state'] = 'Anomalous-high risk'
                
                    
                    # deletar os numDataJump ultimos numeros da lista de dados, pois foram caracterizados como anomalos
                    # (nao servirao como treino/teste de proximos dados)
                    if( (msg['IA']['AI_state'].startswith('Anomalous')) or (msg['IA']['AI_state'].endswith('risk')) ):
                        msg['IA']['count'] = msg['IA']['count']-numDataJump
                        msg['payload']['temp']['anoms'] = msg['payload']['temp']['anoms']+numDataJump
                        for varname in varnamelist:
                            for _ in range(numDataJump):
                                msg['IA'][varname].pop()
                    	
                    msg['IA']['AI_error'] = scored_train['Loss_mae'].mean()
                    msg['IA']['AI_tolerance'] = msg['payload'][nomerede]['tolerance']
                    #del scored_train,Xtrain_t,X_pred,model,ark,X_train,scaler,train, test_scaler, li
                    gc.collect()
        #fim do teste

        # Avalia se 500 novos dados foram inseridos, indica que a esta rede passivel de treino
        if(msg['IA']['count']%500==0):
            # treino!
            #nome_ia = msg['IA']['nome']
            #nomerede = msg['IA']['nomerede']
            li = [msg['IA']['PPDG'],msg['IA']['TPDG'],msg['IA']['PTPT'],msg['IA']['TTPT'],msg['IA']['PPT']]
            li = np.asarray(li)
            li = li.transpose()
            train = pd.DataFrame(data = li, columns= ["PPDG", "TPDG", "PTPT", "TTPT", "PPT"])
            for key in train.keys():
                train[key] = train[key].fillna(np.nanmean(np.array(train[key])))
            seed(10)
                
            # # load, average and merge sensor samples
            if(len(train)<1):
                print('Dados Insuficientes')
            else:
                train = train.reset_index(drop=True)
                X_train = scaler.fit_transform(train)
                
                # reshape inputs for LSTM [samples, timesteps, features]
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            
                # fit the model to the data
                history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size).history
                #network train success                    
                net_success= history["loss"][len(history["loss"])-1]
                if(np.isnan(net_success)):
                    continue

                # plot the loss distribution of the training set
                X_pred = model.predict(X_train)
                X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                X_pred = pd.DataFrame(X_pred, columns=train.columns)
                X_pred.index = train.index
            
                scored_train = pd.DataFrame(index=train.index)
                Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
                scored_train['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
            
                # save all model information, including weights, in h5 format
                ark = model.get_weights()
                
                scaler_output = {}
                for scalerkeys in scaler.__dict__:
                    scaler_output[scalerkeys] = getattr(scaler, scalerkeys)
                    if(isinstance(scaler_output[scalerkeys],np.ndarray)):
                        scaler_output[scalerkeys] = scaler_output[scalerkeys].tolist()
                scaler_output = json.dumps([scaler_output])

                if(('payload' not in msg) or (msg['payload'] is None)):
                    msg['payload'] = {}
                if((nomerede not in msg['payload']) or (msg['payload'][nomerede] is None) or  len(msg['payload'][nomerede]) == 0):
                    msg['payload'][nomerede] = {}
                msg['payload'][nomerede]['weights'] = pd.Series(ark).to_json(orient='values')
                msg['payload'][nomerede]['scaler'] = scaler_output #pd.Series(scaler_output).to_json(orient='values')
                msg['payload'][nomerede]['tolerance'] = (scored_train['Loss_mae'].mean() + 3*scored_train['Loss_mae'].std())
                
                if(('log_treino' not in msg['payload']) or (msg['payload']['log_treino'] is None) or  len(msg['payload']['log_treino']) == 0):
                    msg['payload']['log_treino'] = []
                msg['payload']['log_treino'].append(j + 1 + msg['payload']['temp']['auxnum'])
                
                # para apagar, temporario
                msg['payload']['temp']['treinos'] = msg['payload']['temp']['treinos']+1
            gc.collect()
        #fim do treino

msg['payload']['temp']['auxnum'] = j + 1 + msg['payload']['temp']['auxnum']

# apaga os dados ja computados
msg['data'][well_name]['IA'] = {}

msg['topic'] = 'UFAL_ANOMALY_' + nome_IA + '_IA'
return msg