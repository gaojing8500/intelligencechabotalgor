# Introduction 
* 意图识别其实一个多分类问题，可以采用分类模型做，也可以采用意图识别+槽位填充联合深度学习模型(rasa nlu diet classifiers)
### 使用工具
* sklearn 中 Multinomial Naive Bayes Classifier,KNN Classifier,Logistic Regression Classifier,Random Forest Classifier,Decision Tree Classifier,GBDT(Gradient Boosting Decision Tree) Classifier,SVM Classifier
* 神经网路分类模型：Fasttext(监督语料) FastCNN FastRNN,RCNN.Transformer(hanlp)等 