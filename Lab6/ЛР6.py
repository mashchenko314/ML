import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv("data.csv")
    return data


TEST_SIZE = 0.3
RANDOM_STATE = 0


def preprocess_data(data):
    data.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data = data.drop(columns=['fractal_dimension_mean', 'texture_se', 'symmetry_se'])
    data_X = data.drop(columns='diagnosis')
    data_y = data['diagnosis']
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(data_X, \
                                                                            data_y, test_size=TEST_SIZE, \
                                                                            random_state=RANDOM_STATE)
    return data_X_train, data_X_test, data_y_train, data_y_test


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


# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}


@st.cache(suppress_st_warning=True)
def print_models(models_select, X_train, X_test, y_train, y_test):
    current_models_list = []
    roc_auc_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:,1]
   
        roc_auc = roc_auc_score(y_test.values, Y_pred_proba)
        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        #Отрисовка ROC-кривых 
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
        plot_confusion_matrix(model, X_test, y_test.values, ax=ax[1],
                        display_labels=['0','1'], 
                        cmap=plt.cm.Blues, normalize='true')
        fig.suptitle(model_name)
        st.pyplot(fig)

    if len(roc_auc_list)>0:
        temp_d = {'roc-auc': roc_auc_list}
        temp_df = pd.DataFrame(data=temp_d, index=current_models_list)
        st.bar_chart(temp_df)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

st.header('ЛР №6. Создание веб-приложения для демонстрации моделей машинного обучения.')
data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)
if st.checkbox('Показать данные'):
    st.write(data.head())
if st.checkbox('Показать корреляционную матрицу'):
    fig_corr, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig_corr)

st.header('Оценка качества моделей')
print_models(models_select, X_train, X_test, y_train, y_test)