import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




#畫圖程式
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
        # setup markers generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
    
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 =  np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
        z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
        
        # hightlight test samples
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')
 
    
    
 #主程式
def main():
    iris = datasets.load_iris()
    
    
    x_train, x_test, y_train, y_test = train_test_split(iris.data[:, [2, 3]], iris.target, test_size=0.25, random_state=4)
    
    
    #decisiontree
    model_dtc = DecisionTreeClassifier() 
    model_dtc.fit(x_train,y_train)
    y_pred3 = model_dtc.predict(x_test)
    print('Test Set Score:{:.2f} decisiontree'.format(np.mean(y_test == y_pred3)))
    
   
    #svm
    model_svc = SVC()                      
    model_svc.fit(x_train,y_train)          
    y_pred4 = model_svc.predict(x_test) 
    plt.scatter(x_train[:,0],x_train[:,1], c=y_train)
    plt.scatter(x_test[:,0],x_test[:,1], c=y_pred4)  
    print('Test Set Score:{:.2f} svc'.format(np.mean(y_test == y_pred4)))
    
    
    #KNN
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=1, p=2, weights='uniform')
    knn = KNeighborsClassifier(n_neighbors=1)#設置為最近鄰=1
    knn.fit(x_train,y_train)
    y_pred2 = knn.predict(x_test)
    X,y = datasets.load_iris(True)
    lc = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF']) #颜色
    lc2 = ListedColormap(['#FF0000','#00FF00','#0000FF'])
 
    
    plt.scatter(x_test[:,0],x_test[:,1],c = y_pred2,cmap = lc)
    plt.scatter(X[:,0],X[:,1],c = y,cmap = lc2)
    print('Test Set Score:{:.2f} KNN' .format(knn.score(x_test,y_test)))
    
    
    #logistic
    X = iris.data[:, :2]  
    Y = iris.target
    logreg = LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    #點出上下邊界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    #把資料放到圖形上查看分類結果
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired) 
    y_pred = logreg.predict(x_test)
    print('Test Set Score:{:.2f} logistic'.format(np.mean(y_test == y_pred)))
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length, Logistic ')
    plt.ylabel('Sepal width')
    #羅吉斯的圖
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.show()
    



    X_combined = np.vstack((x_train, x_test))
    y_combined = np.hstack((y_train, y_test))
    
    #把各種顯算法的圖畫出來
    #svm
    plot_decision_regions(X_combined, y_combined, classifier=model_svc, test_idx=range(105, 150))
    plt.xlabel('petal length [cm],SVC')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    #knn
    plot_decision_regions(X_combined, y_combined, classifier=knn, test_idx=range(105, 150))
    plt.xlabel('petal length [cm],Knn')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    #decision_tree
    plot_decision_regions(X_combined, y_combined, classifier=model_dtc, test_idx=range(105, 150))
    plt.xlabel('petal length [cm],Dicision Tree')
    plt.ylabel('petal width [cm]')
    plt.legend(loc='upper left')
    plt.show()
    #邏輯司徒在上方
    
    


main()
