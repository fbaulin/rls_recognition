# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 19:07:33 2019
Модуль выполняет обучение классификатора
@author: eXabicher
"""
# Библиотеки для Database
import itertools as it
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.linear_model import SGDRegressor
from skfeature.function.information_theoretical_based import MRMR
from sklearn.neural_network import MLPRegressor
from skfeature.function.similarity_based import fisher_score
# from sklearn.model_selection import GridSearchCV as GreedSCV  # библиотека для перебора
# Библиотеки общего назначения
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as GradBstCfr
from sklearn.ensemble import RandomForestClassifier as RndFrstCfr
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler as StdSclr
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as ppl
import datetime
from sklearn.utils.multiclass import unique_labels


class Dataset:
    """ Класс для работы с обучающей выборкой.

        Attributes:
            feature_header:     названия признаков.
            features:           матрица, содержащая признаки.
            targets_vect(np.array):  one-hot целевые векторы.
            targets:            массив идентификаторов.
            class_names:        уникальные имена классов.
            filename(str):      имя файла базы векторов признаков.

    """

    def __init__(self, filename=None):
        """ Конструктор объекта класса.

        Args:
            filename(str): название файла.
        
        """
        self.feature_header = None  # заголовок признаков
        self.features = None        # векторы признаков
        self.targets_vect = None    # one-hot целевые векторы
        self.targets = None         # идентификаторы классов
        self._names_ids = None      # словарь имя->идентификатор
        self.class_names = None     # уникальные имена классов
        if filename is not None:
            self.filename = filename    # имя файла базы векторов признаков
            self.load_csv(filename)
        else:
            self.filename = ''

    def load_data(self, x, y):
        """ Сохраняет данные в объект, выделяет уникальные имена классов.
        
        Args:
            x: векторы признаков, расположенные по строкам.
            y: названия классов.

        """
        self.features = x
        self.targets = y
        self.class_names = unique_labels(y)

    def load_csv(self, filename):
        """ Загрузить данные из .csv файла базы векторов признаков.

        Args:
            filename(str): имя csv файла, в котором содержится выборка.

        """
        # input format: [ f_1, f_2, ..., f_n, t ], header id included in csv
        in_data = pd.read_csv(filename)                                 # загрузить csv с данными
        self.feature_header = in_data.columns[:-1]                      # извлечь заголовок таблицы
        self.features = in_data.values[:, :-1].astype(np.float64)       # извлечь значения признаков
        self.targets, unique_names = pd.factorize(in_data.values[:, -1], sort=1)  # получить id и имена классов
        self.targets_vect = np.array(pd.get_dummies(self.targets))      # сформировать one-hot векторы
        self._names_ids = dict(zip(
            unique_names, range(len(unique_names))      # сформировать словарь имя-id
        ))
        self.class_names = unique_names                 # записть в объект уникальные имена классов

    def get_fisher_scores(self, max_dim):
        """ Получить меру Фишера и качество распознавания на основе AUC ROC.

        Выполняется отбор признаков для размерностей пространства признаков от 1 до max_dim. Для каждой размерности
        выполняется перекрестная проверка (cross-validation) и вычисляется интегральное значение меры Фишера и
        среднее по всем подвыборкам значение меры AUC ROC.

        Args:

            max_dim(int): число признаков до которого следует производить отбор.

        Returns:

            fisher_summary_scores: - вычисленные суммарные значения меры Фишера.
            auc_roc_scores: - вычисленные значения площади под кривой ROC.

        """

        x_train = scale(self.features)                          # normalize features
        y_train = self.targets                                  # target ids
        # Fisher score estimation
        f_score = fisher_score.fisher_score(x_train, y_train)   # calculate Fisher score value
        ranked_f_score = fisher_score.feature_ranking(f_score)  # rank features
        print('Последовательность отобранных коэффициентов:')
        print(*list(self.feature_header[ranked_f_score[0:max_dim]]), sep=', ')
        fisher_summary_scores = list(it.accumulate(f_score[ranked_f_score[0:max_dim]]))     # integral Fisher scores
        # Cross validation
        k_fold = KFold(n_splits=5, shuffle=True)                # setup cross-validation pattern
        ar_scorer = make_scorer(roc_auc_score)                  # make scorer
        clf = SGDRegressor(max_iter=100, tol=1e-3, random_state=241)  # stochastic gradient descend regression as a clf
        auc_roc_scores = []                                     # list for AUC ROC values
        for i in range(1, max_dim + 1):  # iterate by number of features selected
            features = x_train[:, ranked_f_score[0:i]]          # select features
            t = y_train
            vect_auc_roc_score = cross_val_score(clf, features, t, scoring=ar_scorer, cv=k_fold)  # train
            auc_roc_scores.append(np.mean(vect_auc_roc_score))  # add mean (over CV-subsets) AUC ROC value

        return fisher_summary_scores, auc_roc_scores

    def get_mrmr_score(self, max_dim):
        """ Итеративно выполняет отбор признаков на основе меры "minimal redundancy maximum relevance" (MRMR).

            Args:
                max_dim(int): предельное число признаков, которые следует отобрать.

        """

        x_train = scale(self.features)    # feature normalization
        y_train = self.targets                  # targets vector

        kf = KFold(n_splits=5, shuffle=True, random_state=241)  # make CV tool
        ar_scorer = make_scorer(roc_auc_score)                  # make scorer tool
        clf = MLPRegressor(hidden_layer_sizes=(20, 10))         # multilayer perceptron as a classifier
        auc_roc_scores = []
        for n_features in range(1, max_dim+1):
            mrmr_idx, _, _ = MRMR.mrmr(x_train, y_train, n_selected_features=n_features)
            features = x_train[:, mrmr_idx]
            vect_auc_roc_score = cross_val_score(clf, features, y_train, scoring=ar_scorer, cv=kf)  # train
            auc_roc_scores.append(np.mean(vect_auc_roc_score))  # save mean value of auc roc on CV

        return auc_roc_scores


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          graphic=False,
                          title=None,
                          cmap=ppl.cm.get_cmap('Blues')):
    """ Вывести рисунок с матрицами ошибок.
    Нормированные матрицы могут быть получены.

    Args:
        y_true:     целевые векторы.
        y_pred:     векторы, предсказанные алгоритмом.
        classes:    задать вручную имена классов для подписи.
        normalize:  нормированные/абсолютные значения ошибок.
        graphic:    вывести на дисплей.
        title:      задать вручную значение заголовка.
        cmap:       задать вречную цвета матрицы.

    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(classes)
    print(cm)

    if graphic:
        if title is None:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'
        fig, ax = ppl.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        ppl.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax


# вывод информации
def data_out(llval_train, llval_test, lr):
    """ Вывод графической информации - рисование графиков .
        
    Args:
        llval_train.
        llval_test.
        lr: скорость обучения.

    """
    train_length = len(llval_train)     #
    test_length = len(llval_test)       #
    ppl.figure()
    ppl.plot(range(1, train_length+1), llval_train, 'r')    # график ошибки на обучающей выборке
    ppl.plot(range(1, test_length+1), llval_test, 'b:')     # график ошибки на проверочной выборке
    ppl.title('Learn rate= ' + str(lr))                     #
    ppl.show()
    print('Min val test_n')
    print(str(np.min(llval_test)) + ' ' + str(np.argmin(llval_test) + 1))


# расчет вероятности с помощью сигмоидальной функции
def c_sigm(y_pred):
    return 1 / (1 + np.exp(-y_pred))


# получение функции ошибок (кросэнтропийные потери) для gradboost на курсе Я
def get_loss_staged(x, y, clf):
    # передается обученный классификатор
    n_estimators = clf.n_estimators         # число базовых алгоритмов
    llval = np.zeros(n_estimators)          # подготовить контейнер под оценки
    loss = []                       # контейнер с решениями
    pred = clf.staged_decision_function(x)
    for p in pred:                  # перевод полученных результатов в вероятности черерз сигмойду
        loss.append(c_sigm(p))
    for i in range(n_estimators):   # оценка кросс-энтропии
        llval[i] = log_loss(np.array(pd.get_dummies(y)), loss[i])
    return llval


# подбор оптимальной скорости обучения градиентного бустинга по кросс-энтропии
def setup_gradboost(x_set, y_set, n_estimators=50):
    # инициализация

    learn_rates = [1, 0.4, 0.2, 0.1]    # скорости обучения
    llval_train = np.zeros([len(learn_rates), n_estimators])
    llval_test = np.zeros([len(learn_rates), n_estimators])

    # оценить качество работы
    xtr, xts, ytr, yts = tts(x_set, y_set, test_size=0.3, random_state=241)  # разбиение выборки
    for i_lr in range(len(learn_rates)):
        lr = learn_rates[i_lr]
        # инициализация и обучение классификатора
        print('Lerning rate: ' + str(lr))
        clf = GradBstCfr(n_estimators=n_estimators, verbose=True, random_state=241, learning_rate=lr)
        clf.fit(xtr, ytr)
        llval_train[i_lr, :] = get_loss_staged(xtr, ytr, clf)   # расчет функции потерь на обучающей выборке
        llval_test[i_lr, :] = get_loss_staged(xts, yts, clf)    # расчет функции потерь на проверочной выборке
        data_out(llval_train[i_lr, :], llval_test[i_lr, :], lr)  # вывод информации
        print()


#
def setup_random_forrest(x_set, y_set):
    # инициализация
    n_estimators = 200  # число базовых алгоритмов
    # настройки исходных данных

    xtr, xts, ytr, yts = tts(x_set, y_set, test_size=0.7, random_state=241)  # разбиение выборки

    tclf = RndFrstCfr(random_state=241, n_estimators=n_estimators)  # init classifier with # of trees
    tclf.fit(xtr, ytr)
    tllos = log_loss(yts, tclf.predict_proba(xts))
    print('For tree log-los= ' + str(tllos))


# сравнительная оценка классификаторов
def clf_assessment(clf, xtr=None, ytr=None, xts=None, yts=None, class_names=None,
                   clf_name=None, fspace='fs',
                   png=False):
    """
    Оценка классификатора: вычисление перекрестной энтропии и матрицы ошибок.

    Args:
        clf: классифиувтор.
        xtr: обучающая выборка.
        ytr: целевой вектор обучающей выборки.
        xts: тестовый выборка.
        yts: целевой вектор тестовой выборки.
        class_names: имена классов для подписи строк и столюцов матрицы.
        clf_name: названия классификатора для заголовка рисунка.
        fspace: название признаков для заголовка рисунка.
        png: сохранить матрицу в файл и продолжить: True - сохранить png, False - показать результат.

    Returns:
        матрица ошибок.
    """
    if clf_name is None:
        clf_name = str(clf).split('(')[0]
    if (xtr is not None) and (ytr is not None):
        clf.fit(xtr, ytr)
    tllos = log_loss(yts, clf.predict_proba(xts))
    print(f'For {clf_name}:\n cross entropy: ' + str(tllos))
    print(f' f1 score: ' + str(f1_score(yts, clf.predict(xts), average='macro')))
    plot_confusion_matrix(yts, clf.predict(xts), classes=class_names, normalize=True, graphic=True,
                          title=f'{clf_name}({fspace})')
    if png:
        ppl.savefig(f'{fspace} {clf_name}.png')
        ppl.close()
    else:
        ppl.show()
    return clf


# сравнение классификаторов для РЛС "Коломна"
def clf_compare(dataset=None, t=None, filename=None,
                n_estimators_tree=80, n_estimators_gbc=30):

    if type(filename) == str:
        print(datetime.datetime.now().strftime('%H:%M:%S'), f'Загрузка')
        dataset = Dataset()  # формирование реализации обучающей выборки
        print(datetime.datetime.now().strftime('%H:%M:%S'), f'Обработка')
        dataset.load_csv(filename)  # открыть файл csv

    if type(dataset) == Dataset:
        print('Набор данных ('+dataset.filename+') загружен из пространства')
    elif (type(dataset) == np.ndarray) and t is not None:
        dataset = Dataset()
        dataset.load_data(dataset, t)

    xtr, xts, ytr, yts = tts(dataset.features, dataset.targets, test_size=0.2)  # разбиение выборки
    sscaler = StdSclr()
    xtr = sscaler.fit_transform(xtr)
    xts = sscaler.transform(xts)

    clfs = []

    clf = RndFrstCfr(n_estimators=n_estimators_tree)  # init classifier with # of trees
    clfs.append(clf_assessment(clf, xtr, ytr, xts, yts, class_names=dataset.class_names,
                   clf_name='RandForrest', png=True))

    clf = GradBstCfr(n_estimators=n_estimators_gbc, learning_rate=0.4)
    clfs.append(clf_assessment(clf, xtr, ytr, xts, yts, class_names=dataset.class_names,
                   clf_name='GradBoost', png=True))

    clf = SVC(kernel='rbf', gamma='scale',
              verbose=False, probability=True, decision_function_shape='ovr')
    clfs.append(clf_assessment(clf, xtr, ytr, xts, yts, class_names=dataset.class_names,
                   clf_name='SVC', png=True))

    return clfs, sscaler


def main():
    print(datetime.datetime.now().strftime('%H:%M:%S'), 'Пакет экспериментов Коломна')
    print(f'--------- Начало секции  ------------')
    # ml_ds = Dataset()
    # ml_ds.load_csv(filename='dataset/kolomna_sp.csv')
    # print(ml_ds.get_mrmr_score(8))
    clf_compare(filename='dataset/kolomna_sp.csv')
    print(f'--------- Конец секции   ------------')
    print(datetime.datetime.now().strftime('%H:%M:%S'), 'Пакет экспериментов Коломна выполнен ')


if __name__ == '__main__':
    main()

