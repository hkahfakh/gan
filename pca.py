from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from getData import get_data, data_split

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def knn_classifier(train_X, train_y, test_X, test_y):
    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(train_X, train_y)
    result = knn.predict(test_X)
    return accuracy_score(result, test_y), metrics.recall_score(result, test_y, average='micro')

# 返回降维后的特征
def dimensionReduction(X,dim=10):
    pca = PCA(n_components=dim)
    # X_pca:降维后的X
    X_pca = pca.fit_transform(X)
    print(pca.noise_variance_)
    return X_pca

if __name__ == '__main__':
    data = get_data("./dataSet/UNSW_finally.npy")
    X, y = data[:, :-1], data[:, -1]
    X_pca = dimensionReduction(X,10)
    X_train,X_test,y_train,y_test = train_test_split(X_pca, y, test_size=0.2)
    print(knn_classifier(X_train, y_train, X_test, y_test))
