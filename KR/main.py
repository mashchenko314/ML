import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from supervised.automl import AutoML
from tpot import TPOTClassifier


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.5, a - 0.05, str(round(b, 3)), color='white')
        plt.show()


def load_data():
    # Загрузка данных
    data = pd.read_csv('BankChurners.csv')
    return data

# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, 
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


# функции для обучения моделей
def train_model(model_name, model, classMetricLogger, is_print=1):
    model.fit(X_train, Y_train)
    # Предсказание значений
    Y_pred = model.predict(X_test)

    precision = precision_score(Y_test.values, Y_pred)
    recall = recall_score(Y_test.values, Y_pred)
    f1 = f1_score(Y_test.values, Y_pred)
    roc_auc = roc_auc_score(Y_test.values, Y_pred)

    classMetricLogger.add('precision', model_name, precision)
    classMetricLogger.add('recall', model_name, recall)
    classMetricLogger.add('f1', model_name, f1)
    classMetricLogger.add('roc_auc', model_name, roc_auc)
    fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
    draw_roc_curve(Y_test.values, Y_pred, ax[0])
    plot_confusion_matrix(model, X_test, Y_test.values, ax=ax[1],
                      display_labels=['0','1'], 
                      cmap=plt.cm.Blues, normalize='true')
    fig.suptitle(model_name)
    if is_print == 1:
        st.pyplot(fig)

    if is_print == 1:
        st.write(f'--------------------{model_name}--------------------')
        st.write(model)
        st.write(f"precision_score: {precision}")
        st.write(f"recall_score: {recall}")
        st.write(f"f1_score: {f1}")
        st.write(f"roc_auc: {roc_auc}")
        st.write(f'--------------------{model_name}--------------------\n')


data = load_data()

#Производим удаление последних двух столбцов, как указано в описании к данному датасету
data.drop(columns=data.columns[[data.shape[1]-2, data.shape[1]-1]], inplace=True)

st.sidebar.header('Логистический регрессор')
cs_1 = st.sidebar.slider('Параметр регуляризации:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Модель ближайших соседей')
n_estimators_2 = st.sidebar.slider('Количество K:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('SVC')
cs_3 = st.sidebar.slider('Регуляризация:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Дерево решений')
max_depth_4 = st.sidebar.slider('Максимальная глубина:', min_value=10, max_value=50, value=10, step=1)

st.sidebar.header('Случайный лес')
n_estimators_5 = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)

st.sidebar.header('Градиентный бустинг')
n_estimators_6 = st.sidebar.slider('Количество:', min_value=6, max_value=15, value=6, step=1)

st.header('Курсовой проект по анализу данных')
st.write('В качестве набора данных мы будем использовать данные об оттоке клиентов в банке, выдающем кредитные карты - https://www.kaggle.com/sakshigoyal7/credit-card-customers.')
st.write('В рассматриваемом примере будем решать задачу классификации. Для решения задачи классификации в качестве целевого признака будем использовать "Attrition_Flag ". Поскольку признак содержит только значения 2 значения: Existing Customer (1) и Attrited Custome (0), то это задача бинарной классификации.')

# Первые пять строк датасета
st.subheader('Первые 5 значений')
st.write(data.head())

st.subheader('Размер датасета')
st.write(data.shape)

st.subheader('Колонки и их типы данных')
st.write(data.dtypes)

st.subheader('Наличие пустых значений')
st.write(data.isnull().sum())
st.write('Вывод. Представленный набор данных не содержит пропусков.')

st.subheader('Статистические данные')
st.write(data.describe())

st.subheader('Уникальные значения для категориальных признаков')
col_obj = data.dtypes[data.dtypes==object].index.values.tolist()
for i in enumerate(col_obj):
    uniq_obj = data[i[1]].unique()
    st.write(f'{i[0]+1}. {i[1]}: {uniq_obj} | КОЛ-ВО: {len(uniq_obj)}')

st.subheader('Оценка дисбаланса классов для Attrition_Flag')
fig1 = plt.figure(figsize=(3,3))
ax = plt.hist(data['Attrition_Flag'])
st.pyplot(fig1)

# посчитаем дисбаланс классов
total = data.shape[0]
class_0, class_1 = data['Attrition_Flag'].value_counts()
st.write('Класс Existing Customer составляет {}%, а класс Attrited Custome составляет {}%.'
      .format(round(class_0 / total, 4)*100, round(class_1 / total, 4)*100))

st.write('Вывод. Дисбаланс классов присутствует.')


st.subheader('Кодирование категориальных признаков')
st.write('Закодируем признак Gender и целевой признак Attrition_Flag на основе подхода LabelEncoding. Для остальных категориальных признаков используем метод One-hot Encoding')
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
mapping = {'Existing Customer' : 0, 'Attrited Customer' : 1}
data['Attrition_Flag'] = data['Attrition_Flag'].apply(lambda x: mapping[x])

#Скопируем датасет для применения к его категориальным признакам метода 'Label encoding' с целью удобства дальнейшего представления корреляционной матрицы, а также последующего применения при использовании классификатора дерева решений
data_copy = data
data_copy = data_copy.apply(LabelEncoder().fit_transform)


data = pd.get_dummies(data, drop_first=True)


st.write('Убедимся, что целевой признак для задачи бинарной классификации содержит только 0 и 1.')
st.write(data['Attrition_Flag'].unique())

st.subheader('Корреляционная матрица')
fig1, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data_copy.corr())
st.pyplot(fig1)

data_all=data.copy()

st.subheader('Масштабирование данных')
st.write('Масштабирование выполняем на основе Z-оценки.')
scale_cols = data.dtypes[data.dtypes!=object].index.values.tolist()
scale_cols.remove('Attrition_Flag')
scale_cols.remove('Gender')
se = StandardScaler()
sc1_data = se.fit_transform(data[scale_cols])
data[scale_cols] = se.fit_transform(data[scale_cols])

scaled_cols = data_copy.dtypes[data_copy.dtypes!=object].index.values.tolist()
scaled_cols.remove('Attrition_Flag')
scaled_cols.remove('Gender')
data_copy[scaled_cols] = se.fit_transform(data_copy[scaled_cols])

# Добавим масштабированные данные в набор данных
for i in range(len(scale_cols)):
    col = scale_cols[i]
    new_col_name = col + '_scaled'
    data_all[new_col_name] = sc1_data[:, i]

st.subheader('Проверим, что масштабирование не повлияло на распределение данных')
for col in scale_cols:
    col_scaled = col + '_scaled'
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(data_all[col], 50)
    ax[1].hist(data_all[col_scaled], 50)
    ax[0].title.set_text(col)
    ax[1].title.set_text(col_scaled)
    st.pyplot(fig)


st.subheader('Корреляционная матрица после масштабирования')
fig1, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(data_copy.corr())
st.pyplot(fig1)


st.subheader('Проведение корреляционного анализа данных. Формирование промежуточных выводов о возможности построения моделей машинного обучения.')
st.write('На основе корреляционной матрицы можно сделать следующие выводы:')
st.write('1. Корреляционные матрицы для исходных и масштабированных данных совпадают.')
st.write('2. Среди двух сильно коррелирующих признаков "Credit_Limit"  и "Avg_Open_To_Buy" необходимо удалить признак "Avg_Open_To_Buy", так как он сильнее коррелирует с другими признаками объектов и слабее коррелирует с целевым признаком')
st.write('3. Среди двух сильно коррелирующих признаков "Total_Trans_Amt"  и "Total_Trans_Ct" необходимо удалить признак "Total_Trans_Amt", так как он сильнее коррелирует с другими признаками объектов')

data_copy.drop(columns=['Avg_Open_To_Buy', 'Total_Trans_Amt'], inplace=True)
data.drop(columns=['Avg_Open_To_Buy', 'Total_Trans_Amt'], inplace=True)

st.write(data.head())

# разделение выборки на обучающую и тестовую
# X_train, X_test, Y_train, Y_test, X, Y = preprocess_data(data)
X = data.drop("Attrition_Flag", axis=1)
Y = data["Attrition_Flag"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

st.subheader('Выбор метрик для последующей оценки качества моделей.')
st.write('В качестве метрик для решения задачи классификации будем использовать:')
st.write('1. Метрика precision')
st.write('2. Метрика recall (полнота)')
st.write('3. Метрика F1-мера')
st.write('4. Метрика ROC AUC')

st.subheader('Выбор метрик для последующей оценки качества моделей.')
st.write('В качестве метрик для решения задачи классификации будем использовать:')
st.write('1. Логистическая регрессия')
st.write('2. Метод ближайших соседей')
st.write('3. Метод опорных векторов')
st.write('4. Решающее дерево')
st.write('5. Случайный лес')
st.write('6. Градиентный бустинг')

st.subheader('Обучим модели')
# Модели
models = {'LogR': LogisticRegression(C=cs_1),
          'KNN': KNeighborsClassifier(n_neighbors=n_estimators_2),
          'SVC': SVC(C=cs_3, probability=True),
          'Tree': DecisionTreeClassifier(max_depth=max_depth_4, random_state=10),
          'RF': RandomForestClassifier(n_estimators=n_estimators_5, oob_score=True, random_state=10),
          'GB': GradientBoostingClassifier(n_estimators=n_estimators_6, random_state=10)}

# Сохранение метрик
classMetricLogger = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLogger)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader('Сравнение метрик моделей')
# Метрики качества модели
metrics = classMetricLogger.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее подобранное значение параметра регуляризации для логистической регрессии:')
params = {'C': np.logspace(1, 3, 20)}
grid_lr = GridSearchCV(estimator=LogisticRegression(),
                       param_grid=params,
                       cv=3,
                       n_jobs=-1)
grid_lr.fit(X_train, Y_train)
st.write(grid_lr.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'LogR': LogisticRegression(C=cs_1),
          'LogRGrid': grid_lr.best_estimator_}

# Сохранение метрик
classMetricLoggerLogR = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerLogR)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerLogR.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerLogR.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества ближайших соседей для модели ближайших соседей:')
params = {'n_neighbors': list(range(5, 100, 5))}
grid_knn = GridSearchCV(estimator=KNeighborsClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_knn.fit(X_train, Y_train)
st.write(grid_knn.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'KNN': KNeighborsClassifier(n_neighbors=n_estimators_2),
          'KNNGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerKNN = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerKNN)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerKNN.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerKNN.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение параметра регуляризации для SVC модели')
params = {'C': np.logspace(1, 3, 20)}
grid_svc = GridSearchCV(estimator=SVC(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_svc.fit(X_train, Y_train)
st.write(grid_svc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'SVC': SVC(C=cs_3, probability=True),
          'SVCGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerSVC = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerSVC)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerSVC.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerSVC.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение максимальной глубины для дерева решений:')
params = {'max_depth': list(range(5, 500, 10))}
grid_dtc = GridSearchCV(estimator=DecisionTreeClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_dtc.fit(X_train, Y_train)
st.write(grid_dtc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'Tree': DecisionTreeClassifier(max_depth=max_depth_4, random_state=10),
          'TreeGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerTree = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerTree)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerTree.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerTree.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества фолдов для случайного леса:')
params = {'n_estimators': list(range(5, 200, 10))}
grid_rfc = GridSearchCV(estimator=RandomForestClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_rfc.fit(X_train, Y_train)
st.write(grid_rfc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'RF': RandomForestClassifier(n_estimators=n_estimators_5, oob_score=True, random_state=10),
          'RFGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerRF = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerRF)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerRF.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerRF.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Лучшее значение количества фолдов для градиентного бустинга:')
params = {'n_estimators': list(range(5, 200, 10))}
grid_gbc = GridSearchCV(estimator=GradientBoostingClassifier(),
                        param_grid=params,
                        cv=3,
                        n_jobs=-1)
grid_gbc.fit(X_train, Y_train)
st.write(grid_gbc.best_params_)

st.subheader('Сравним с baseline-моделью')
# Модели
models = {'GB': GradientBoostingClassifier(n_estimators=n_estimators_6, random_state=10),
          'GBGrid': grid_knn.best_estimator_}

# Сохранение метрик
classMetricLoggerGB = MetricLogger()
for model_name, model in models.items():
    train_model(model_name, model, classMetricLoggerGB)
    train_model(model_name, model, classMetricLogger, 0)

# Метрики качества модели
metrics = classMetricLoggerGB.df['metric'].unique()
# Построим графики метрик качества модели
for metric in metrics:
    st.pyplot(classMetricLoggerGB.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

st.subheader('Сравнение метрик для всех моделей')
metrics = classMetricLoggerGB.df['metric'].unique()
# Построим графики метрик качества всех моделей
for metric in metrics:
    st.pyplot(classMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6)))

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

st.subheader('Модель обученная с помощью TPOT')

tpot = TPOTClassifier(generations=5, population_size=50,cv=5, verbosity=2, random_state=1)
tpot.fit(X_train, Y_train)
st.write(tpot.score(X_test, Y_test))
tpot.export('tpot_digits_pipeline.py')



