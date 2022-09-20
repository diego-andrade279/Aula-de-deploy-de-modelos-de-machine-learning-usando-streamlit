
import pickle as pkl
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn import metrics
import shap 

## Criação do Pipeline de predição do modelo
def pipeline_predict(df, obj):

    if obj == 'Predição de Churn':

        # Feature Engineering
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes

        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes

        df['area_code'] = df['area_code'].astype('category')
        df['area_code'] = df['area_code'].cat.codes

        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        # Leitura do modelo treinado usando pickle 
        with open('xgb_model', 'rb') as files:
            model_trained = pkl.load(files)

        # Pega as features usadas no modelo
        features = model_trained.get_boosted().feature_names
        df = df[features] # filtra as features usadas no modelo nos dados de entrada

        return model_trained.predict(df), model_trained.predict_proba(df[:,1]) # retorna a predição e a probabilidade


    if obj == 'Explicabilidade' and df == '':

        df = pd.read_csv('churn_train.csv')
        df['churn'] = df['churn'].astype('category')
        df['churn'] = df['churn'].cat.codes
        df['voice_mail_plan'] = df['voice_mail_plan'].astype('category')
        df['voice_mail_plan'] = df['voice_mail_plan'].cat.codes
        df['international_plan'] = df['international_plan'].astype('category')
        df['international_plan'] = df['international_plan'].cat.codes
        df['area_code'] = df['area_code'].astype('category')
        df['area_code'] = df['area_code'].cat.codes
        df['state'] = df['state'].astype('category')
        df['state'] = df['state'].cat.codes

        X = df.drop('churn', axis=1)
        y = df['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7565)

        with open('xgb_model', 'rb') as files:
            model_trained = pkl.load(files)

        # Pega as features usadas no modelo
        features = model_trained.get_boosted().feature_names
        df = df[features] # filtra as features usadas no modelo nos dados de entrada

        prob = model_trained.predict_proba(X_test)[:,1]
        pred = model_trained.predict(X_test)

        # Print das métricas
        st.title('Principais Métricas.')
        st.write("** AUC: **"+str(metrics.roc_auc_score(y_test, prob)))
        st.write("** Accuracy:** "+str(metrics.accuracy_score(y_test, pred)))
        st.write("** Recall:** "+str(metrics.recall_score(y_test, pred)))
        st.write("** F1-Measure:** "+str(metrics.f1_score(y_test, pred)))

        # Grafico de features mais importantes, sempre usar st.pyplot nos graficos
        st.title('Features mais importantes')
        feature_important = model_trained.get_booster().get_score(importance_type='weight')
        keys = list(feature_important.keys())
        values = list(feature_important.values())
        data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=True)
        data.nlargest(40, columns="score").plot(kind='barh', figsize = (20,10))
        st.pyplot()

        st.title('Explicabilidade usando Shap values dos primeiros três registros de teste')
        explainer = shap.Explainer(model_trained)
        shap_values = explainer(X_test)

        shap.plots.waterfall(shap_values[0])
        st.pyplot()
        shap.plots.waterfall(shap_values[1])
        st.pyplot()
        shap.plots.waterfall(shap_values[2])
        st.pyplot()

        shap.summary_plot(shap_values)
        st.pyplot()

        shap.plots.force(shap_values[3])
        st.pyplot()
        shap.plots.force(shap_values[4])
        st.pyplot()
        shap.plots.force(shap_values[5])
        st.pyplot()


