import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier

comments = pd.read_csv("Restaurant-Reviews.csv",on_bad_lines='skip')
nullValues = comments.isnull().sum()
comments = comments.dropna(how="any")
comments.index = range(704);
nullValues2 = comments.isnull().sum()

# PREPROCESSING 
ps = PorterStemmer()
nltk.download('stopwords') 
compileComments = []
for i in range(704):
    comment = re.sub('[^a-zA-Z]',' ',comments['Review'][i]) 
    comment = comment.lower() 
    comment = comment.split() 
    comment = [ps.stem(word) for word in comment if not word in set(stopwords.words('english'))]
    comment = ' '.join(comment)
    compileComments.append(comment)
    #print(comment)


# Feature Extraction 
# Bag of Words (BOW)
cv = CountVectorizer(max_features=2000) 
X = cv.fit_transform(compileComments).toarray()  
y = comments.iloc[:,1].values  


# Machine Learning
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def accuracyScoreGraph(cm,title,score):  
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('{0} Accuracy Score: {1}'.format(title,score), size = 15)
    plt.show()

# Predictions by machine learning algorithms
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
score = gnb.score(x_test, y_test) 
cm = confusion_matrix(y_test, y_pred) 
cr = classification_report(y_test,y_pred) 
print("GaussianNB\n",classification_report(y_test, y_pred))
accuracyScoreGraph(cm,"GaussianNB",score)



lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred2 = lr.predict(x_test)    
score2 = lr.score(x_test, y_test)
cm2 = confusion_matrix(y_test, y_pred2)
print("LogisticRegression\n",classification_report(y_test, y_pred2))
accuracyScoreGraph(cm2,"LogisticRegression",score2)


knc = KNeighborsClassifier(n_neighbors=1,weights="distance",algorithm="ball_tree",metric="minkowski",p=2,leaf_size=30)
knc.fit(x_train, y_train)
y_pred3 = knc.predict(x_test)
cm3 = confusion_matrix(y_test,y_pred3)
score3 = knc.score(x_test, y_test)
print("KNeighborsClassifier\n",classification_report(y_test, y_pred3))
accuracyScoreGraph(cm3,"KNN",score3)


dtc = DecisionTreeClassifier(random_state=42,criterion="entropy",splitter="random",max_features="auto") 
dtc.fit(x_train, y_train)
y_pred4 = dtc.predict(x_test)
cm4 = confusion_matrix(y_test, y_pred4)
score4 = dtc.score(x_test, y_test)
print("DecisionTreeClassifier\n",classification_report(y_test, y_pred4))
accuracyScoreGraph(cm4,"DecisionTreeClassifier",score4)

# The best hyperparameter estimation for the model using GridSearchCV
dtc_params = {'criterion':("gini", "entropy"),'splitter':("best", "random"),
              'max_depth': range(3,13), 'max_features': range(5,15)}
dtc_grid = GridSearchCV(dtc, dtc_params, cv=5, n_jobs=-1,verbose=True)
dtc_grid.fit(x_train,y_train)
print(dtc_grid.best_params_)
print('Best cv mean result: {0}'.format(round(dtc_grid.best_score_,5)))
print('Best holdout result: {0}'.format(round(accuracy_score(y_test, dtc_grid.predict(x_test)),5)))


svc = SVC(kernel="rbf",gamma="scale",cache_size=100,decision_function_shape="ovr") 
svc.fit(x_train, y_train)
y_pred5 = svc.predict(x_test)
score5 = svc.score(x_test, y_test)
cm5 = confusion_matrix(y_test, y_pred5)
print("SVC\n",classification_report(y_test, y_pred5))
accuracyScoreGraph(cm5,"SVC",score5)

# The best hyperparameter estimation for the model using GridSearchCV
svc = SVC()
svc_params = {'kernel':('linear', 'rbf'),
              'gamma':('scale', 'auto'),
              'decision_function_shape':('ovo', 'ovr'),
              'C': [10, 100]}
clf = GridSearchCV(svc,svc_params)
clf.fit(x_train,y_train)
print(clf.best_params_)  
print('Best cv mean result: {0}'.format(round(clf.best_score_,5)))
print('Best holdout result: {0}'.format(round(accuracy_score(y_test, clf.predict(x_test)),5)))


rfc = RandomForestClassifier(n_estimators=8,criterion="entropy",max_depth=20, max_features="auto",class_weight="balanced_subsample",random_state=0) 
rfc.fit(x_train, y_train)
y_pred6 = rfc.predict(x_test)
score6 = rfc.score(x_test, y_test)
cm6 = confusion_matrix(y_test, y_pred6)
print("RandomForestClassifier\n",classification_report(y_test, y_pred6))
accuracyScoreGraph(cm6,"RandomForestClassifier",score6)


X = pd.DataFrame(X,index=range(0,704),columns=range(0,1133))
plt.figure(figsize=(16, 9))
ranking = rfc.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns
plt.title("Feature importances based on Random Forest Classifier", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="aqua", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()


print(cross_val_score(RandomForestClassifier(n_estimators=8, 
                                  random_state=0, max_features= "auto",class_weight="balanced_subsample", 
                            max_depth= 20,criterion="entropy"), 
                              x_train, y_train))
print('mean of cv-scores: {0}'.format(round(np.mean(cross_val_score(RandomForestClassifier(n_estimators=8, 
                                  random_state=0, max_features= "auto",class_weight="balanced_subsample",
                            max_depth= 20,criterion="entropy"), 
                              x_train, y_train)),4)))
# The best hyperparameter estimation for the model using GridSearchCV
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, 
                                random_state=0)

rf_params = {'max_depth': range(3,13), 'max_features': range(5,15)}

rf_grid = GridSearchCV(rf, rf_params,
                           cv=5, n_jobs=-1, 
                       verbose=True)

rf_grid.fit(x_train, y_train)
print(rf_grid.best_params_)
print('Best cv mean result: {0}'.format(round(rf_grid.best_score_,5)))
print('Best holdout result: {0}'.format(round(accuracy_score(y_test, rf_grid.predict(x_test)),5)))

