import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import base64

# Define image paths
image_path1 = r"C:\Users\ALLTEC\Desktop\stramlit\images\Map-of-Beja-Tunisia-showing-location-of-farms-where-raw-bovine-milk-samples-were.png"
image_path2 = r"C:\Users\ALLTEC\Desktop\stramlit\images\Capture d’écran (427).png"
image_path3 = r"C:\Users\ALLTEC\Desktop\stramlit\images\sp\download.png"
image_path4 = r"C:\Users\ALLTEC\Desktop\stramlit\images\sp\random 13.png"
image_path5 = r"C:\Users\ALLTEC\Desktop\stramlit\images\sp\xgb1.png"
image_path6 = r"C:\Users\ALLTEC\Desktop\stramlit\images\sp\light1.png"
image_path7 = r"C:\Users\ALLTEC\Desktop\stramlit\images\sp\svm1.png"

image_path11 = r"C:\Users\ALLTEC\Desktop\stramlit\images\p\lin reg1.png"
image_path12 = r"C:\Users\ALLTEC\Desktop\stramlit\images\p\random1.png"
image_path13 = r"C:\Users\ALLTEC\Desktop\stramlit\images\p\xgb1.png"
image_path14 = r"C:\Users\ALLTEC\Desktop\stramlit\images\p\light1.png"
image_path15 = r"C:\Users\ALLTEC\Desktop\stramlit\images\p\svm1.png"
# Load images
image1 = Image.open(image_path1)
image2 = Image.open(image_path2)
image3 = Image.open(image_path3)
image4 = Image.open(image_path4)
image5 = Image.open(image_path5)
image6 = Image.open(image_path6)
image7 = Image.open(image_path7)

image11 = Image.open(image_path11)
image12 = Image.open(image_path12)
image13 = Image.open(image_path13)
image14 = Image.open(image_path14)
image15 = Image.open(image_path15)

# Function to encode image to base64 for HTML embedding
def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Function to display images side by side in Streamlit
def display_images():
    col1, col2 = st.columns(2)
    with col1:
        st.image(image1, caption='Position géographique de la zone étudiée en Béja, Tunisie', use_column_width=True)
    with col2:
        st.image(image2, caption='La répartition spatiale des échantillons de sol', use_column_width=True)

# Function to display variable distribution
def display_variable_distribution(data):
    st.title("Distribution des variables")
    selected_columns = st.multiselect("Sélectionner les variables", data.columns)
    if st.button("Afficher la distribution"):
        for column in selected_columns:
            st.subheader(f"Distribution de {column}")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(data[column], kde=True, ax=ax)
            st.pyplot(fig)

def main():
    # Display logo and user type in sidebar
    image_path = r"C:\Users\ALLTEC\Desktop\stramlit\images\logo1.png"
    image_base64 = get_image_base64(image_path)
    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
            <img src="data:image/png;base64,{image_base64}" style="width: 100px; height: 100px; border-radius: 50%;">
            <h3 style="text-align: center;">User</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.markdown("All rights are reserved © 2024")
    
    # Load data and handle file uploader in main body
    st.write("<h2>Télécharger votre fichier Excel ici:</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Choisir fichier Excel", type=["xlsx"])
    if file:
        data = pd.read_excel(file)
        st.success("Fichier chargé avec succès !")

        # Dashboard title and statistics
        st.write("# Dashboard de la matière organique SOC")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Max. SOC", data['Organic Carbon(g/kg soil)'].max())
        a2.metric("Min. SOC", data['Organic Carbon(g/kg soil)'].min())
        a3.metric("Count. SOC", data['Organic Carbon(g/kg soil)'].count())
        a4.metric("Mean. SOC", data['Organic Carbon(g/kg soil)'].mean())

        # Display images
        display_images()

        # Display variable distribution
        if file.name =="Données sols (3).xlsx":
          display_variable_distribution(data)

        # Check if file name is "111111dataset_uniform_filled P.xlsx"
        if file.name == "111111dataset_uniform_filled P.xlsx" :
                        st.title("Sélectionner le modèle de machine learning")         
                        selected_model = st.multiselect("Sélectionner le modèle", ["Linear Regression", "Random Forest", "SVM", "LightGBM", "XGBoost" ])
                        if st.button("Évaluation des approches with k-cross validation"):
                            st.title("Évaluation des approches with k-cross validation sans pollution")
                            datawp = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_evaluation_resultswithout (1).xlsx")
                            st.bar_chart(datawp)
                            st.title("Évaluation des approches with k-cross validation pollution")
                            datap = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_evaluation_resultspollution.xlsx")
                            st.bar_chart(datap)
                        for model in selected_model:
                            if model == "Linear Regression":
                                st.title("Evaluation des données utilisant Linear regression")
                                c1,c2 = st.columns(2)
                                with c1:
                                    st.image(image11, caption='Modéle de "Linear regression sans pollution"', use_column_width=True)
                                with c2:
                                    datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultsLRP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsLRP (1).xlsx" )
                                st.bar_chart(dataev)
                            elif model == "Random Forest":
                                st.title("Evaluation des données utilisant Random forest")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image12, caption='Modéle de "Random forest sans pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultsrandomP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsRFP (1).xlsx" )
                                st.bar_chart(dataev)
                            elif model == "SVM":
                                st.title("Evaluation des données utilisant SVM")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image15, caption='Modéle de "SVM sans pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultsSVMP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsSVMP (1).xlsx" )
                                st.bar_chart(dataev)
                            elif model == "LightGBM":
                                st.title("Evaluation des données utilisant LightGBM")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image14, caption='Modéle de "LightGBM sans pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultslgbmP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsLRP (1).xlsx" )
                                st.bar_chart(dataev)
                            elif model == "XGBoost":
                                st.title("Evaluation des données utilisant XGBoost")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image13, caption='Modéle de "XGBoost sans pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultsxgbP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsxgb.xlsx" )
                                st.bar_chart(dataev)
        if file.name == "111111dataset_uniform_filled SP.xlsx":

            st.title("Sélectionner le modèle de machine learning")
            selected_model = st.multiselect("Sélectionner le modèle", ["Linear Regression", "Random Forest", "SVM", "LightGBM", "XGBoost"])
            if st.button("Évaluation des approches with k-cross validation"):
                            st.title("Évaluation des approches with k-cross validation sans pollution")
                            datawp = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_evaluation_resultswithout (1).xlsx")
                            st.bar_chart(datawp)
                            st.title("Évaluation des approches with k-cross validation pollution")
                            datap = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_evaluation_resultspollution.xlsx")
                            st.bar_chart(datap)
            
            # Add code to train and evaluate selected models
            for model in selected_model:
                if model == "Linear Regression":
                    st.title("Evaluation des données utilisant Linear regression")
                    c1,c2 = st.columns(2)
                    with c1:
                        st.image(image3, caption='Modéle de "Linear regression sans pollution"', use_column_width=True)
                    with c2:
                        datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\comparisonlinear (2).xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metrics SP.xlsx" )
                    st.bar_chart(dataev)
                elif model == "Random Forest":
                    st.title("Evaluation des données utilisant Random forest")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image4, caption='Modéle de "Random forest sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison random_Results.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsrandom.xlsx" )
                    st.bar_chart(dataev)
                elif model == "SVM":
                    st.title("Evaluation des données utilisant SVM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image7, caption='Modéle de "SVM sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_ResultsSVM.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsSVM.xlsx" )
                    st.bar_chart(dataev)
                elif model == "LightGBM":
                    st.title("Evaluation des données utilisant LightGBM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image6, caption='Modéle de "LightGBM sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_Resultslgbm.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsLightGBM.xlsx" )
                    st.bar_chart(dataev)
                elif model == "XGBoost":
                    st.title("Evaluation des données utilisant XGBoost")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image5, caption='Modéle de "XGBoost sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\Comparison_Resultsxgb.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel(r"C:\Users\ALLTEC\Desktop\stramlit\dataset\model_metricsxgb.xlsx" )
                    st.bar_chart(dataev)

            # Add code to display results
            # ...

            # You can add more code here to handle the selected models




if __name__ == "__main__":
    main()