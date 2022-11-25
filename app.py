import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def main():
      st.title("Data Analysis and Data Visualization")
      st.text("Using Machine Learning and Streamlit")
      activity = ['Data Analysis','Data Visualization','Model Building','About']
      choice = st.sidebar.selectbox("Select your Activity",activity)
      # Code for Data Analysis
      if choice  == 'Data Analysis':
            st.subheader("Explore your Data")

            data = st.file_uploader("Upload Dataset of  CSV format",type=["csv"])
            if data is not None:
                  if st.checkbox("Show preview of Dataset"):
                        df = pd.read_csv(data)
                        st.dataframe(df.head())

                  if st.checkbox("Show shape"):
                        st.text("(rows,columns)")
                        st.write(df.shape)

                  if st.checkbox("Show Columns"):
                        all_columns = df.columns.to_list()
                        st.write(all_columns)

                  if st.checkbox("Select columns to Show"):
                        selected_columns = st.multiselect("Select Columns",all_columns)
                        new_df = df[selected_columns]
                        st.dataframe(new_df)

                  if st.checkbox("Show Summary"):
                        st.write(df.describe())

                  if st.checkbox("Show Value Count"):
                        st.write(df.iloc[:,1].value_counts())










            
       #Code for Data Visualization
      elif choice  == 'Data Visualization':
            st.subheader("Visualize your Data")

            data = st.file_uploader("Upload Dataset",type=["csv","txt","xls","xlsx"])
            if data is not None:
                  df = pd.read_csv(data)
                  st.dataframe(df.head())

            if st.checkbox("Heat Map Chart"):
                  st.write(sns.heatmap(df.corr(),annot = True))
                  st.pyplot()

            if st.checkbox("Pie Chart"):
                  all_columns = df.columns.to_list()
                  column_to_plot = st.selectbox("Select 1 column to plot",all_columns)
                  pie_plot = df[column_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
                  st.write(pie_plot)
                  st.pyplot()

            all_columns_names = df.columns.tolist()
            type_of_plot = st.selectbox("Select type of plot",["area","bar","line","box"])
            selected_columns_names = st.multiselect("Select columns to plot",all_columns_names)

            if st.button("Generate Plot"):
                  st.success("Generating customize plot of {} for {}".format(type_of_plot,selected_columns_names))

                  if type_of_plot == 'area':
                        cust_data = df[selected_columns_names]
                        st.area_chart(cust_data)

                  elif type_of_plot == 'bar':
                        cust_data = df[selected_columns_names]
                        st.bar_chart(cust_data)
                        
                  elif type_of_plot == 'line':
                        cust_data = df[selected_columns_names]
                        st.line_chart(cust_data)
                        
                  elif type_of_plot == 'area':
                        cust_data = df[selected_columns_names]
                        st.area_chart(cust_data)

                 


            
       #Code for ML Model Building
      elif choice  == 'Model Building':
            st.subheader("Model your Data with ML")

            data = st.file_uploader("Upload Dataset",type=["csv","txt","xls","xlsx"])
            if data is not None:
                  df = pd.read_csv(data)
                  st.dataframe(df.head())

                  X = df.iloc[:,0: -1]
                  Y = df.iloc[:,-1]
                  seed = 7

                  models = []
                  models.append(("Logistic Regression",LogisticRegression()))
                  models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
                  models.append(("K Nearest Neighbors",KNeighborsClassifier()))
                  models.append(("Decission Tree",DecisionTreeClassifier()))
                  models.append(("Naive Bayes",GaussianNB()))
                  models.append(("Support Vector Machine",SVC()))


                  model_names = []
                  model_mean = []
                  model_std = []
                  all_models = []
                  scoring = 'accuracy'

                  for name,model in models:
                        kfold = model_selection.KFold(n_splits = 10,random_state=seed)
                        cv_results  = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring = scoring)
                        model_names.append(name)
                        model_mean.append(cv_results.mean())
                        model_std.append(cv_results.std())

                        accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
                        all_models.append(accuracy_results)

                  if st.checkbox("Matrics as table"):
                        st.dataframe(pd.DataFrame(zip(model_names,model_mean,model_std),columns = ["Model Name","Model Accuracy","Standard Deviation"]))
                  if st.checkbox("Matrics as JSON"):
                        st.json(all_models)








      elif choice  == 'About':
            st.subheader("About")
            st.header("This is an open source app build to automate Data Analysis,Data Visualization and Machine Learning Model Building")
            st.header("Hope all dev's using this app find it interesting and useful. Kindly suggest Ideas for improvement and contribute to this App")
            st.header("Creator GitHub:  https://github.com/gouravojha/Web_Apps")
            st.header("Co-Creator GitHub: https://github.com/AshutoshGeek")




if  __name__ == '__main__':
      main()
      
      
