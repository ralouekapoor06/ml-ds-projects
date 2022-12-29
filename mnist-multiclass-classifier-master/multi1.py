from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

#we load the mnist dataset
r=fetch_mldata('MNIST original')

#split the data into X and y
X=r['data']
y=r['target']

#print the shape
print(X.shape)
print(y.shape)
#we can see that the shape of X is 70000,780 i.e there are 70000 images and
#each image has 784 pixels (784 features in this case)

#we need to split the data into trainset and the testset
#We only work on the trainset until the model is final
train_x,test_x,train_y,test_y=X[:60000],X[60000:],y[:60000],y[60000:]

#also we should shuffle the images randomly.
#there might be cases when a number of same images are together,it doesnt works properly
shuffle=np.random.permutation(60000)
train_x,train_y=train_x[shuffle],train_y[shuffle]

#Also we must scale the training images for better results
scalar=StandardScaler()
train_x=scalar.fit_transform(train_x.astype(np.float64))

#we use a SGDClassifier
#random state is set because SGD can then give reproducible results
#by default we use a OVA classifier but we can use an OVO classifier
#the OVA classifier takes one digit and runs the model by taking all the remaining
#digits against it----(one vs all)
#In case of OVO classifier we just take 2 digits at a time and run the model
#the data required for each model is quite less but the number of models are high
#SVM uses this method
#OVO for this dataset will have to run n*(n-1)/2 classifiers
sgd=SGDClassifier(random_state=42)

#we then pass in train_x and train_y into the fit method
#the fit method can generalize the results which can be used for predictions
sgd.fit(train_x,train_y)

#we then calculate the accuracy
#we split the training data into 3 folds as cv=3 then we train any 2 parts
#the 3rd part gives the accuracy
#so we get 3 accuracies in this case
score=cross_val_score(sgd,train_x,train_y,cv=3,scoring="accuracy")
print(score)
#The accuracy is pretty decent which is above 90% for all the 3 parts

#We know that accuracy is not always the best metric to evaluate on
#So we use percision and recall
#for that we use the confusion matrix
#we can also understand the model well with the help of precision and recall
#the no.of rows gives the number of classes
#the number of classes specifies the number fields we are doing our classification
#on
#suppose we have a 2 x 2 matrix then the top left and bottom right positions are
#specified as true negatives and true positives and we want more of that
#we also need to calculate the train_y_pred before proceeding
train_y_pred=cross_val_predict(sgd,train_x,train_y,cv=3)
confmatrix=confusion_matrix(train_y,train_y_pred)

#also lets see the plot
plt.matshow(confmatrix,cmap=plt.cm.gray)
plt.show()
#we can see that most of the white spots are in the middle and hence its a good thing!
#In the middle although we can see that there is a dark block which means
#that column number has fewer images or it is not fully classified

#Lets go along each row and sum it up and get a new matrix norm_conf
#we also fill the diagonals with 0 as we want to just focus on the errors
row_sums=confmatrix.sum(axis=1,keepdims=True)
norm_conf=confmatrix/row_sums
np.fill_diagonal(norm_conf,0)
plt.matshow(norm_conf,cmap=plt.cm.gray)
plt.show()
#if the columns of a particular number is bright that means images are getting misclassified with that number
#the rows which are bright have gotten confused with other digits
#we should spend time improving the classification of 8 and 9 as well as 3 and 5 confusion needs to be sorted because of the
#lighter color

#lets finally get the precision score and the recall score
#we also calculate the f1_score which is the harmonic mean of precision_score
#and recall_score
prescore=precision_score(train_y,train_y_pred,average='micro')
recallscore=recall_score(train_y,train_y_pred,average='micro')
f1score=f1_score(train_y,train_y_pred,average='micro')
print(prescore)
print(recallscore)
print(f1score)
#we can see that the score is above 91% which is good
