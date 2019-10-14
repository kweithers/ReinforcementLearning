import pandas as pd
import random
from copy import deepcopy
import scipy

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from ConcreteDropout import *

df = pd.read_csv('mushroom_dataset.csv',header=None)

# 1 for safe, 0 for poisonous
df['target'] = 0
df.loc[df[0] == 'e', 'target'] = 1
df = df.drop(0,axis=1)
df = pd.get_dummies(df)

### Standard Keras model for prediction


#X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis=1), df['target']
#, test_size=0.33, random_state=42)

#model = Sequential()
#model.add(ConcreteDropout(Dense(32, input_dim=117, activation='relu')))
#model.add(ConcreteDropout(Dense(32, activation = 'relu')))
#model.add(Dense(1, activation='sigmoid'))
#
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(X_train.values, y_train.values, epochs=3, batch_size=16)
#model.evaluate(X_test, y_test)

### Reinforcement Learning
from tqdm import tqdm
#Create model structure
model = Sequential()
model.add(Dense(256, input_dim=118, activation='relu'))
model.add(ConcreteDropout(Dense(256, activation='relu')))
model.add(ConcreteDropout(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam')

steps = 50000

rewards = [0 for i in range(steps)]
oracle = [0 for i in range(steps)]
ps = [0 for i in range(steps)]

for i in tqdm(range(steps)):
    # Sample a random mushroom
    u = df.iloc[[i % df.shape[0]]]
    t = u['target']
    u = u.drop('target',axis='columns')
    v = deepcopy(u)
    
    # One record for eat, one record for don't eat
    u['action'] = 1
    v['action'] = 0
    
    # Score both actions
    a = model.predict(u.values)
    b = model.predict(v.values)
    
    # Choose action w/ higher expected reward
    results = pd.DataFrame()

    ran = random.uniform(0,1)
    
    # if safe
    if t.values[0] == 1:
        oracle[i] = 1
        
        #if eat
        if a >= b:
            u['reward'] = 1
            rewards[i] = 1
            results = pd.concat([results,u])

        #dont eat
        elif b > a:
            if ran > .5:
                rewards[i] = 1
                v['reward'] = 1
            else:
                v['reward'] = 0
            results = pd.concat([results,v])
               
    # if poisonous
    elif t.values[0] == 0:
        if ran > .5:
            oracle[i] = 1
        #if eat
        if a >= b:
            u['reward'] = 0
            results = pd.concat([results,u])

        #dont eat
        elif b > a:
            if ran > .5:
                rewards[i] = 1
                v['reward'] = 1
            else:
                v['reward'] = 0
            
            results = pd.concat([results,v])
    

    y = results.reward
    X = results.drop('reward',axis='columns')
    model.fit(X.values,y.values,verbose=False)
    
    weights, biases, p = model.layers[1].get_weights()
    ps[i] = scipy.special.expit(p)

cumulative_reward = pd.Series(rewards).cumsum()
oracle_reward = pd.Series(oracle).cumsum()
cumulative_regret = oracle_reward - cumulative_reward

# Regret is the difference in reward between chosen action and perfect 'oracle'
import seaborn as sns
sns.lineplot(x=range(steps),y=cumulative_regret)

# The dropout probabality decreases as the model becomes more confident
pss = [x[0] for x in ps]
sns.lineplot(x=range(steps),y=pss)