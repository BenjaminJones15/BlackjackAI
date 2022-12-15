import numpy 
import pandas 
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression



dataset = pandas.read_csv(os.path.basename('../blkjckhands.csv'))
#here is the data that would be visible to a player plus the hand outcome
blackjackhands = dataset[['card1','card2','sumofcards','dealcard1','winloss']].copy()

# first we need to obtain and transform the results of the hands
#0 is loss, 1 is push, 2 is win
le = preprocessing.LabelEncoder()
blackjackhands['winloss'] = le.fit_transform(blackjackhands['winloss'])
target_data = blackjackhands['winloss'].values
training_data = blackjackhands.drop('winloss',axis=1).values
Y=target_data
X=training_data
Xscale = scale(training_data)
Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xscale,Y,test_size=0.10,random_state=None,stratify=None)
#preprocessing is finished

#create and score the model
lr = LogisticRegression()
lr.fit(Xtrain,Ytrain)
y_pred = lr.predict(Xtest)
score = lr.score(Xtest,Ytest)

#  
print(score)



