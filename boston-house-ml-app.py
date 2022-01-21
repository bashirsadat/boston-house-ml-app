import streamlit as st
import pandas as pd
import shap
from PIL import Image
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
image = Image.open('logo.jpg')
cola, colb, colc = st.columns([3,6,1])
with cola:
    st.write("")

with colb:
    st.image(image, width = 300)

with colc:
    st.write("")
menu = ["Home","About"]
choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    st.write("""
    # Boston House Price Prediction App

    This app predicts the **Boston House Price**!
    """)
    st.write('---')

    # Loads the Boston House Price Dataset
    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    Y = pd.DataFrame(boston.target, columns=["MEDV"])
    st.write("""
    **THE DATASET DATA** \n
    Data description \n
    The Boston data frame has 506 rows and 14 columns.This data frame contains the following columns:\n
    crim: per capita crime rate by town,\n
    zn: proportion of residential land zoned for lots over 25,000 sq.ft,\n
    indus: proportion of non-retail business acres per town,\n
    chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise),\n
    nox: nitrogen oxides concentration (parts per 10 million),\n
    rm: average number of rooms per dwelling,\n
    age: proportion of owner-occupied units built prior to 1940,\n
    dis: weighted mean of distances to five Boston employment centres,\n
    rad: index of accessibility to radial highways,\n
    tax: full-value property-tax rate per \$10,000,\n
    ptratio: pupil-teacher ratio by town,\n
    black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town,\n
    lstat: lower status of the population (percent).\n
    """)
    st.write(X)
    st.write("""
    **THE DATASET LABELS**
    medv:median value of owner-occupied homes in \$1000s.
    """)
    st.write(Y)
    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    def user_input_features():
        CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
        ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
        INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
        CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
        NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
        RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
        AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
        DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
        RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
        TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
        PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
        B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
        LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
        data = {'CRIM': CRIM,
                'ZN': ZN,
                'INDUS': INDUS,
                'CHAS': CHAS,
                'NOX': NOX,
                'RM': RM,
                'AGE': AGE,
                'DIS': DIS,
                'RAD': RAD,
                'TAX': TAX,
                'PTRATIO': PTRATIO,
                'B': B,
                'LSTAT': LSTAT}
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_features()

    # Main Panel

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')

    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, Y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)

    st.header('Prediction of MEDV')
    st.write(prediction)
    st.write('---')

    # Explaining the model's predictions using SHAP values
    # https://github.com/slundberg/shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.header('Feature Importance')
    f, ax = plt.subplots()
    ax.set_title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X)
    st.pyplot(f,bbox_inches='tight')
    st.write('---')

    f, ax = plt.subplots()
    ax.set_title('Feature importance based on SHAP values (Bar)')
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(f,bbox_inches='tight')
else:
    st.subheader("About")
    st.write("With a hybrid profile of data science and computer science, I’m pursuing a career in AI-driven firms. I believe in dedication, discipline, and creativity towards my job, which will be helpful in meeting your firm's requirements as well as my personal development.")
    st.write("Check out this project's [Github](https://github.com/bashirsadat/boston-house-ml-app)")
    st.write(" My [Linkedin](https://www.linkedin.com/in/saadaat/)")
    st.write("See my other projects [LinkTree](https://linktr.ee/saadaat)")
