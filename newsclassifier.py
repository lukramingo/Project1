import pandas as pd
import numpy as np
import re
from textblob import Word

import os

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer

# to splite dataset into train and test data.
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

# importing Machine learning Algorithm from sklearn
from sklearn.linear_model import LogisticRegression #Logistic Regression
from sklearn.linear_model import LinearRegression #Linear Regression
from sklearn.neighbors.nearest_centroid import NearestCentroid # Centroid Based Classifier
from sklearn.neighbors import KNeighborsClassifier #KNN Classifier
from sklearn.svm import SVC # Support Vector Machine
from sklearn.naive_bayes import BernoulliNB # Naive Bayes Classifier
from sklearn.tree import DecisionTreeClassifier # Decission Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

data_folder = "./datasets"
folders = ["business","entertainment","politics","sport","tech"]

os.chdir(data_folder)

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names,)
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r')
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlabel('Predicted label')
    plt.ylabel('Actual label')
    return fig 

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    """
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"\'ve", "", string)
    string = re.sub(r"n\'t", "", string)
    string = re.sub(r"\'re", "", string)
    string = re.sub(r"\'d", "", string)
    string = re.sub(r"\'ll", "", string)
    string = re.sub(r",", "", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", "", string)
    string = re.sub(r"\)", "", string)
    string = re.sub(r"\?", "", string)
    string = re.sub(r"'", "", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[0-9]\w+|[0-9]","", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# x = []
# y = []

# for i in folders:
# 	files = os.listdir(i)
# 	for text_file in files:
# 		file_path = i + "/" +text_file
# 		# print ("reading file:", file_path)
# 		with open(file_path, encoding="utf8", errors='ignore') as f:
# 			data = f.readlines()
# 		data = ' '.join(data)
# 		x.append(data) # adding text to the list
# 		y.append(i)	   # adding corresponding label

# data = {'news': x, 'type': y}
# row_data = pd.DataFrame(data)
# row_data.to_csv('../mydatset.csv', index=False)

data = pd.read_csv('../mydatset.csv')
# print ("BERORE  ::  ", data['news'][0])
from sklearn import preprocessing
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(data['type'])
data['type'] = labelEncoder.transform(data['type'])
# # print(data.head())

for index,value in enumerate(data['news']):
    # print "processing data:",index
    data['news'][index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])

# print ("\nAFTER  ::  ", data['news'][0])

vect = TfidfVectorizer(stop_words='english',min_df=2)
feature = vect.fit_transform(data["news"])
# print(vect.get_feature_names())
# feature_matrix = feature.todense
# data1 ={"feature": feature, "label": data["type"]}
# df = pd.DataFrame(data1)
# df.to_csv('../feature.csv', index=False)
label = data["type"]
# Suffle data to increase accuracy
label, feature = shuffle(label, feature, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.20, random_state=0)

# # Now train the model
linearRegr = LinearRegression().fit(x_train, y_train)
logisticRegr = LogisticRegression().fit(x_train, y_train)
model_centroid = NearestCentroid().fit(x_train, y_train)
model_knn = KNeighborsClassifier(28).fit(x_train, y_train)
model_svm = SVC().fit(x_train, y_train)
model_nb = BernoulliNB().fit(x_train, y_train)
model_dtree = DecisionTreeClassifier(criterion = "entropy",
 random_state = 100, max_depth=3, min_samples_leaf=5).fit(x_train, y_train)
model_rfc = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1).fit(x_train, y_train)

# # # Find the accuracy of each model
phrase = "The accuracy of %s is %0.2f"
accu_lir = linearRegr.score(x_test, y_test)
print(phrase % ("Linear Regression", 100*accu_lir))
accu_lr = logisticRegr.score(x_test, y_test)
print(phrase % ("Logistic Regression", 100*accu_lr))
accu_centroid = model_centroid.score(x_test, y_test)
print(phrase % ("Centroid Based Classifier", 100*accu_centroid))
accu_knn = model_knn.score(x_test, y_test)
print(phrase % ("KNN", 100*accu_knn))
accu_svm = model_svm.score(x_test, y_test)
print(phrase % ("SVM", 100*accu_svm))
accu_nb = model_nb.score(x_test, y_test)
print(phrase % ("Naive Bayes", 100*accu_nb))
accu_dtree = model_dtree.score(x_test, y_test)
print(phrase % ("Decission Tree", 100*accu_dtree))
accu_rfc = model_rfc.score(x_test, y_test)
print(phrase % ("RandomForest Classifier", 100*accu_rfc))
y_pred = model_centroid.predict(x_test)


#CROSS VALIDATION AND FINDING BEST K VALUE OF KNN
from sklearn.model_selection import cross_val_score
k_scores = []
for i in range(10,51):
    kn = KNeighborsClassifier(i)
    scores =cross_val_score(kn, feature, label, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
print("k value with best accuracy is ", np.where(np.array(k_scores) == np.array(k_scores).max())[0])

plt.plot(k_scores)
plt.show()
##########################################################

print("Confusion Matrix")
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("Method 1 (Seaborn)")
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(accu_lr)
plt.title(all_sample_title, size = 15);

plt.figure = print_confusion_matrix(cm,folders)


plt.show()

true_pos = np.diag(cm)
print("True Positive : ",true_pos)

false_pos = np.sum(cm, axis=1) - true_pos
print("False Positive : ",false_pos)

false_neg = np.sum(cm, axis=0) - true_pos
print("False Negative : ",false_neg)

precision = np.sum(np.diag(cm) / np.sum(cm, axis=0))/len(np.diag(cm))
print("\nPrecision : ",precision)
recall = np.sum(np.diag(cm) / np.sum(cm, axis=1))/len(np.diag(cm))
print("Recall : ",recall)
print("F1 Score : ", (2*precision*recall)/(precision+recall))

y_pred = logisticRegr.predict(x_test)

scores = cross_val_score(logisticRegr, feature, label, cv=5)
print("\n Cross Validation Score \n")
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print(classification_report(y_test,y_pred))

def predictNews(news_article):
	tf_idf_news = vect.transform([news_article])
	# print (tf_idf_news)
	print("The given news type is ",logisticRegr.predict(tf_idf_news))


article = """Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (?600m) for the three months to December, from $639m year-earlier.

The firm, which is now one of the biggest investors in Google, benefited from sales of high-speed internet connections and higher advert sales. TimeWarner said fourth quarter sales rose 2% to $11.1bn from $10.9bn. Its profits were buoyed by one-off gains which offset a profit dip at Warner Bros, and less users for AOL.

Time Warner said on Friday that it now owns 8% of search-engine Google. But its own internet business, AOL, had has mixed fortunes. It lost 464,000 subscribers in the fourth quarter profits were lower than in the preceding three quarters. However, the company said AOL's underlying profit before exceptional items rose 8% on the back of stronger internet advertising revenues. It hopes to increase subscribers by offering the online service free to TimeWarner internet customers and will try to sign up AOL's existing customers for high-speed broadband. TimeWarner also has to restate 2000 and 2003 results following a probe by the US Securities Exchange Commission (SEC), which is close to concluding.

Time Warner's fourth quarter profits were slightly better than analysts' expectations. But its film division saw profits slump 27% to $284m, helped by box-office flops Alexander and Catwoman, a sharp contrast to year-earlier, when the third and final film in the Lord of the Rings trilogy boosted results. For the full-year, TimeWarner posted a profit of $3.36bn, up 27% from its 2003 performance, while revenues grew 6.4% to $42.09bn. "Our financial performance was strong, meeting or exceeding all of our full-year objectives and greatly enhancing our flexibility," chairman and chief executive Richard Parsons said. For 2005, TimeWarner is projecting operating earnings growth of around 5%, and also expects higher revenue and wider profit margins.

TimeWarner is to restate its accounts as part of efforts to resolve an inquiry into AOL by US market regulators. It has already offered to pay $300m to settle charges, in a deal that is under review by the SEC. The company said it was unable to estimate the amount it needed to set aside for legal reserves, which it previously set at $500m. It intends to adjust the way it accounts for a deal with German music publisher Bertelsmann's purchase of a stake in AOL Europe, which it had reported as advertising revenue. It will now book the sale of its stake in AOL Europe as a loss on the value of that stake."""

predictNews(article)