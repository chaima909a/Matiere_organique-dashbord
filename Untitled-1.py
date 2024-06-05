import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px 

# Define image paths
image_path1 = "Map-of-Beja-Tunisia-showing-location-of-farms-where-raw-bovine-milk-samples-were.png"
image_path2 = "Capture d’écran (427).png"
image_path3 = "downloadsp.png"
image_path4 = "random 13sp.png"
image_path5 = "xgb1sp.png"
image_path6 = "light1sp.png"
image_path7 = "svm1sp.png"

image_path11 = "lin reg1.png"
image_path12 = "random1.png"
image_path13 = "xgb1.png"
image_path14 = "light1.png"
image_path15 = "svm1.png"

image_path21 = "lin1 sea.png"
image_path22 = "random sea1.png"
image_path23 = "xgb1 sea.png"
image_path24 = "svm1 sea.png"
image_path25 = "light1 sea.png"
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

image21 = Image.open(image_path21)
image22 = Image.open(image_path22)
image23 = Image.open(image_path23)
image24 = Image.open(image_path24)
image25 = Image.open(image_path25)
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
    image_path = "logo1.png"
    
    st.sidebar.image(image_path, width=100, caption="Utilisateur")
       

    def page_home():
        st.header("Présentation de la zone etudiée Oued Béja, Béja")
        st.subheader("Introduction à Oued Beja")
        st.write("""Oued Beja, située dans le nord-ouest de la Tunisie, est une région d'une richesse historique et culturelle profonde. Nichée entre des paysages montagneux et des vallées fertiles, cette région offre un mélange unique de patrimoine naturel et humain.Connu par ses richesses agricoles, le gouvernorat de Béja se place parmi les premiers gouvernorats dans la production agricole du pays.""")
        datagps = pd.read_excel("donnes gps.xlsx")
        # Charger les données depuis le fichier Excel
        @st.cache_data  # Mettre en cache les données pour éviter de les recharger à chaque rafraîchissement de la page
        def load_data(file_path):
            return pd.read_excel(file_path)

        # Fonction pour afficher la carte
        def display_map(data):
            fig = px.scatter_mapbox(data, lat="Latitude", lon="Longitude", zoom=10)
            fig.update_layout(mapbox_style="open-street-map")
            return fig
        st.subheader("Présentation de l'étude")
        st.write("""Dans le cadre de cette étude, un échantillonnage approfondi a été réalisé dans la région d'Oued Beja, impliquant la collecte de 70 échantillons représentatifs. L'objectif principal de cette campagne d'échantillonnage était de comprendre et de prédire le carbone organique dans cette zone spécifique.""")
        # Chargement des données
        file_path = "donnes gps.xlsx"
        data = load_data(file_path)

        # Affichage de la carte si les données sont disponibles
        if not data.empty:
            st.header("Carte des positions")
            st.write("Affichage des positions à partir du fichier : ", file_path)
            st.plotly_chart(display_map(data))
        else:
            st.error("Le fichier ne contient pas de données ou n'est pas accessible.")
        

    def page_about():
        
        import joblib
        import numpy as np
        # Charger le modèle
        model = joblib.load('model_xgboost.pkl')

        st.title("Prédiction du Carbone Organique du Sol (SOC)")

        # Définir les plages de valeurs pour chaque variable selon les valeurs fournies
        Ph_min, Ph_max = 8.14, 9.18
        Conductivite_min, Conductivite_max = 0.01, 2.702
        Matiere_organique_min, Matiere_organique_max = 1.97, 2.12
        Calcaire_total_min, Calcaire_total_max = 0.012, 0.046
        Calcaire_actif_min, Calcaire_actif_max = 0.103, 0.1425
        Azote_totale_min, Azote_totale_max = 0.0121, 0.05
        Phosphore_assimilable_min, Phosphore_assimilable_max = 3.85, 5.7027
        Potassium_assimilable_min, Potassium_assimilable_max = 31.91, 36.32
        Pb_min, Pb_max = 10.49, 78.3265
        Zn_min, Zn_max = 26.13, 233.07
        Cd_min, Cd_max = 0.05, 1.7
        Cu_min, Cu_max = 1.96, 144.59
        Ni_min, Ni_max = 8.1, 464.67
        B_glucosidase_min, B_glucosidase_max = 33.37, 385.40
        Phosphatase_basique_min, Phosphatase_basique_max = 47.47, 520.23
        Na_min, Na_max = 22.45, 546.00
        K_min, K_max = 4084.57, 9314.52
        Mn_min, Mn_max = 1.11, 3.52
        FDA_min, FDA_max = 1.24, 3.25
        SIR_min, SIR_max = 1.75, 5.30
        CEC_min, CEC_max = 10.24, 14.42

        # Collecter les entrées de l'utilisateur
        Ph = st.slider('Ph', min_value=Ph_min, max_value=Ph_max, value=8.41)
        Conductivite_electrique = st.slider('Conductivité electrique', min_value=Conductivite_min, max_value=Conductivite_max, value=1.0)
        Matiere_organique = st.slider('Teneur en matière organique (M%)', min_value=Matiere_organique_min, max_value=Matiere_organique_max, value=2.0)
        Calcaire_total = st.slider('Calcaire total', min_value=Calcaire_total_min, max_value=Calcaire_total_max, value=0.5)
        Calcaire_actif = st.slider('Calcaire actif', min_value=Calcaire_actif_min, max_value=Calcaire_actif_max, value=0.1)
        Azote_totale = st.slider('Teneur en azote totale', min_value=Azote_totale_min, max_value=Azote_totale_max, value=0.2)
        Phosphore_assimilable = st.slider('Teneur en phosphore assimilable (mg/kg sol)', min_value=Phosphore_assimilable_min, max_value=Phosphore_assimilable_max, value=50.0)
        Potassium_assimilable = st.slider('Teneur en Potassium assimilable (g/kg soil)', min_value=Potassium_assimilable_min, max_value=Potassium_assimilable_max, value=1.5)
        Pb = st.slider('Pb', min_value=Pb_min, max_value=Pb_max, value=10.0)
        Zn = st.slider('Zn', min_value=Zn_min, max_value=Zn_max, value=30.0)
        Cd = st.slider('Cd', min_value=Cd_min, max_value=Cd_max, value=0.1)
        Cu = st.slider('Cu', min_value=Cu_min, max_value=Cu_max, value=20.0)
        Ni = st.slider('Ni', min_value=Ni_min, max_value=Ni_max, value=15.0)
        B_glucosidase = st.slider('B-glucosidase', min_value=B_glucosidase_min, max_value=B_glucosidase_max, value=100.0)
        Phosphatase_basique = st.slider('Phosphatase basique', min_value=Phosphatase_basique_min, max_value=Phosphatase_basique_max, value=50.0)
        Na = st.slider('Na', min_value=Na_min, max_value=Na_max, value=1.0)
        K = st.slider('K', min_value=K_min, max_value=K_max, value=2.0)
        Mn = st.slider('Mn', min_value=Mn_min, max_value=Mn_max, value=1.0)
        FDA = st.slider('FDA', min_value=FDA_min, max_value=FDA_max, value=1.0)
        SIR = st.slider('SIR', min_value=SIR_min, max_value=SIR_max, value=1.0)
        CEC = st.slider('CEC', min_value=CEC_min, max_value=CEC_max, value=10.0)

        # Convertir les entrées en un array numpy
        inputs = np.array([[Ph, Conductivite_electrique, Matiere_organique, Calcaire_total, Calcaire_actif, Azote_totale, Phosphore_assimilable, Potassium_assimilable, Pb, Zn, Cd, Cu, Ni, B_glucosidase, Phosphatase_basique, Na, K, Mn, FDA, SIR, CEC]])

        if st.button("Prédire"):
            prediction = model.predict(inputs)
            st.success(f'La prédiction du Carbone Organique du Sol (SOC) est : {prediction[0]:.2f} g/kg soil')


    def page_contact():
        st.title("Vous pouvez nous contactez ")
        # Collecte des entrées utilisateur
        # Collecte des entrées utilisateur
        # Collecte des entrées utilisateur
        nom = st.text_input("Nom")
        prenom = st.text_input("Prénom")
        email = st.text_input("E-mail")
        message = st.text_input("Message")

        # Créer un bouton "Envoyez"
        if st.button("Envoyez"):
            # Vérifier si tous les champs sont remplis
            if nom and prenom and email and message:
                st.success("Message reçu, nous vous répondrons dans quelques jours. Merci 	:smiley:")
            else:
                st.warning("Veuillez remplir tous les champs avant d'envoyer le message.")

    def main():
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Allez à", ["Home", "About", "Contact"])
        if page == "Home":
            page_home()
        elif page == "About":
            page_about()
        elif page == "Contact":
            page_contact()

    if __name__ == "__main__":
        main()


    st.sidebar.markdown("Tous droits sont réservés © 2024")
    
    # Load data and handle file uploader in main body
    st.write("<h2>Télécharger votre fichier Excel ici:</h2>", unsafe_allow_html=True)
    file = st.file_uploader("Choisir fichier Excel", type=["xlsx"])
    if file:
        data = pd.read_excel(file)
        st.success("Fichier chargé avec succès !")

        # Dashboard title and statistics
        st.write("# Dashboard du carbone organique SOC")
        a1, a2, a3 = st.columns(3)
        a1.metric("Max. SOC", data['Organic Carbon(g/kg soil)'].max())
        a2.metric("Min. SOC", data['Organic Carbon(g/kg soil)'].min())
        a3.metric("Mean. SOC", data['Organic Carbon(g/kg soil)'].mean())

        # Display images
        display_images()

        # Display variable distribution
        if file.name =="Données sols (3) pfe.xlsx":
          display_variable_distribution(data)

        # Check if file name is "111111dataset_uniform_filled P.xlsx"
        if file.name == "111111dataset_uniform_filled P.xlsx" :
                        st.title("Sélectionner le modèle de machine learning")         
                        selected_model = st.multiselect("Sélectionner le modèle", ["Linear Regression", "Random Forest", "SVM", "LightGBM", "XGBoost" ])
                        if st.button("Évaluation des approches with k-cross validation"):
                            st.title("Évaluation des approches with k-cross validation sans pollution")
                            datawp = pd.read_excel("model_evaluation_resultswithout (1).xlsx")
                            st.bar_chart(datawp)
                            st.title("Évaluation des approches with k-cross validation pollution")
                            datap = pd.read_excel("model_evaluation_resultspollution.xlsx")
                            st.bar_chart(datap)
                            st.title("Évaluation des approches with k-cross validation sans enzymes et sans activités microbiennes")
                            datasea = pd.read_excel("model_evaluation_results (2).xlsx")
                            st.bar_chart(datasea)
                        for model in selected_model:
                            if model == "Linear Regression":
                                st.title("Evaluation des données utilisant Linear regression")
                                c1,c2 = st.columns(2)
                                with c1:
                                    st.image(image11, caption='Modéle de "Linear regression avec pollution"', use_column_width=True)
                                with c2:
                                    datalin = pd.read_excel("Comparison_ResultsLRP.xlsx" )
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel("model_metricsLRP (1).xlsx" )
                                st.bar_chart(dataev)
                            elif model == "Random Forest":
                                st.title("Evaluation des données utilisant Random forest")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image12, caption='Modéle de "Random forest avec pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel("Comparison_ResultsrandomP.xlsx")
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel("model_metricsRFP (1).xlsx")
                                st.bar_chart(dataev)
                            elif model == "SVM":
                                st.title("Evaluation des données utilisant SVM")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image15, caption='Modéle de "SVM avec pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel("Comparison_ResultsSVMP.xlsx")
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel("model_metricsSVMP (1).xlsx")
                                st.bar_chart(dataev)
                            elif model == "LightGBM":
                                st.title("Evaluation des données utilisant LightGBM")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image14, caption='Modéle de "LightGBM avec pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel("Comparison_ResultslgbmP.xlsx")
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel("model_metricsLRP (1).xlsx")
                                st.bar_chart(dataev)
                            elif model == "XGBoost":
                                st.title("Evaluation des données utilisant XGBoost")
                                d1,d2 = st.columns(2)
                                with d1:
                                    st.image(image13, caption='Modéle de "XGBoost avec pollution"', use_column_width=True)
                                with d2:
                                    datalin = pd.read_excel("Comparison_ResultsxgbP.xlsx")
                                    st.line_chart(datalin ,width=800)
                                st.write("Evalution du modéle")
                                dataev = pd.read_excel("model_metricsxgb.xlsx")
                                st.bar_chart(dataev)
        if file.name == "111111dataset_uniform_filled SP.xlsx":

            st.title("Sélectionner le modèle de machine learning")
            selected_model = st.multiselect("Sélectionner le modèle", ["Linear Regression", "Random Forest", "SVM", "LightGBM", "XGBoost"])
            if st.button("Évaluation des approches with k-cross validation"):
                            st.title("Évaluation des approches with k-cross validation sans pollution")
                            datawp = pd.read_excel("model_evaluation_resultswithout (1).xlsx")
                            st.bar_chart(datawp)
                            st.title("Évaluation des approches with k-cross validation pollution")
                            datap = pd.read_excel("model_evaluation_resultspollution.xlsx")
                            st.bar_chart(datap)
                            st.title("Évaluation des approches with k-cross validation sans enzymes et sans activités microbiennes")
                            datasea = pd.read_excel("model_evaluation_results (2).xlsx")
                            st.bar_chart(datasea)
            
            # Add code to train and evaluate selected models
            for model in selected_model:
                if model == "Linear Regression":
                    st.title("Evaluation des données utilisant Linear regression")
                    c1,c2 = st.columns(2)
                    with c1:
                        st.image(image3, caption='Modéle de "Linear regression sans pollution"', use_column_width=True)
                    with c2:
                        datalin = pd.read_excel("comparisonlinear (2).xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metrics SP.xlsx")
                    st.bar_chart(dataev)
                elif model == "Random Forest":
                    st.title("Evaluation des données utilisant Random forest")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image4, caption='Modéle de "Random forest sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison random_Results.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsrandom.xlsx")
                    st.bar_chart(dataev)
                elif model == "SVM":
                    st.title("Evaluation des données utilisant SVM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image7, caption='Modéle de "SVM sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_ResultsSVM.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsSVM.xlsx" )
                    st.bar_chart(dataev)
                elif model == "LightGBM":
                    st.title("Evaluation des données utilisant LightGBM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image6, caption='Modéle de "LightGBM sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_Resultslgbm.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsLightGBM.xlsx")
                    st.bar_chart(dataev)
                elif model == "XGBoost":
                    st.title("Evaluation des données utilisant XGBoost")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image5, caption='Modéle de "XGBoost sans pollution"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_Resultsxgb.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsxgb.xlsx")
                    st.bar_chart(dataev)
        if file.name == "111111dataset_uniform_filled sea.xlsx":

            st.title("Sélectionner le modèle de machine learning")
            selected_model = st.multiselect("Sélectionner le modèle", ["Linear Regression", "Random Forest", "SVM", "LightGBM", "XGBoost"])
            if st.button("Évaluation des approches with k-cross validation"):
                            st.title("Évaluation des approches with k-cross validation sans pollution")
                            datawp = pd.read_excel("model_evaluation_resultswithout (1).xlsx")
                            st.bar_chart(datawp)
                            st.title("Évaluation des approches with k-cross validation pollution")
                            datap = pd.read_excel("model_evaluation_resultspollution.xlsx")
                            st.bar_chart(datap)
                            st.title("Évaluation des approches with k-cross validation sans enzymes et sans activités microbiennes")
                            datasea = pd.read_excel("model_evaluation_results (2).xlsx")
                            st.bar_chart(datasea)
            
            # Add code to train and evaluate selected models
            for model in selected_model:
                if model == "Linear Regression":
                    st.title("Evaluation des données utilisant Linear regression")
                    c1,c2 = st.columns(2)
                    with c1:
                        st.image(image21, caption='Modéle de "Linear regression sans enzymes et sans activités microbiennes"', use_column_width=True)
                    with c2:
                        datalin = pd.read_excel("Comparison_ResultsLRPsea.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsseaLR.xlsx")
                    st.bar_chart(dataev)
                elif model == "Random Forest":
                    st.title("Evaluation des données utilisant Random forest")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image22, caption='Modéle de "Random forest sans enzymes et sans activités microbiennes"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_Resultrandomsea.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsseaRF.xlsx")
                    st.bar_chart(dataev)
                elif model == "SVM":
                    st.title("Evaluation des données utilisant SVM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image24, caption='Modéle de "SVM sans enzymes et sans activités microbiennes"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_ResultsSVMsea.xlsx" )
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsseasvm.xlsx" )
                    st.bar_chart(dataev)
                elif model == "LightGBM":
                    st.title("Evaluation des données utilisant LightGBM")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image25, caption='Modéle de "LightGBM sans sans enzymes et sans activités microbiennes"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_Resultslgbmsea.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricssealgbm.xlsx")
                    st.bar_chart(dataev)
                elif model == "XGBoost":
                    st.title("Evaluation des données utilisant XGBoost")
                    d1,d2 = st.columns(2)
                    with d1:
                        st.image(image23, caption='Modéle de "XGBoost sans enzymes et sans activités microbiennes"', use_column_width=True)
                    with d2:
                        datalin = pd.read_excel("Comparison_Resultsxgbsea.xlsx")
                        st.line_chart(datalin ,width=800)
                    st.write("Evalution du modéle")
                    dataev = pd.read_excel("model_metricsseaxgb.xlsx")
                    st.bar_chart(dataev)
            # Add code to display results
            # ...

            # You can add more code here to handle the selected models




if __name__ == "__main__":
    main()
