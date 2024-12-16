from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model

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

li = [msg['IA']['PPDG'],msg['IA']['TPDG'],msg['IA']['PTPT'],msg['IA']['TTPT'],msg['IA']['PPT']]
li = np.asarray(li)
li = li.transpose()
train = pd.DataFrame(data = li, columns= ["PPDG", "TPDG", "PTPT", "TTPT", "PPT"])

# reshape inputs for LSTM [samples, timesteps, features]
train = train.reset_index(drop=True)
X_train = scaler.fit_transform(train)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# fit the model to the data
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size).history
#network train success                    
net_success= history["loss"][len(history["loss"])-1]

# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index
            
scored_train = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)

#------------------------------

li = [msg['IA']['PPDG'],msg['IA']['TPDG'],msg['IA']['PTPT'],msg['IA']['TTPT'],msg['IA']['PPT']]
li = np.asarray(li)
li = li.transpose()
test = pd.DataFrame(data = li, columns= ["PPDG", "TPDG", "PTPT", "TTPT", "PPT"])
test = test.reset_index(drop=True)
X_pred = scaler.transform(test)

# reshape inputs for LSTM [samples, timesteps, features]
X_pred = X_pred.reshape(X_pred.shape[0], 1, X_pred.shape[1])
X_pred = model.predict(X_pred[(len(X_pred)-3):(len(X_pred)-1)])

X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test[0:len(X_pred)].index
                    
scored_train = pd.DataFrame(index=test.index)
X_pred_t = X_pred[(len(X_pred)-3):(len(X_pred)-1)]
X_pred_t = X_pred_t.reshape(X_pred_t.shape[0], X_pred_t.shape[2])
scored_train['Loss_mae'] = np.mean(np.abs(X_pred-X_pred_t), axis = 1)
                    
error_distance = (scored_train['Loss_mae'].mean() / msg['payload'][nomerede]['tolerance'])
                    
if(error_distance <= 1):
	cur_AIstate = 'Normal'
elif(( error_distance > 1)and( error_distance < 1.3 )):
	cur_AIstate = 'Anomalous-low risk'
elif(( error_distance >= 1.3 ) and ( error_distance < 1.7 )):
	cur_AIstate= 'Anomalous-med risk'
elif( error_distance >= 1.7 ):
	cur_AIstate = 'Anomalous-high risk'
