import streamlit as st
import pandas as pd
import json
import requests



def upload_predict_page():
    st.title("Prédire la prime que vous pouvez recevoir")

    st.header("Prédiction des fréquences")
    col1, col2 = st.columns(2)

    with col1:
        Type = st.selectbox(
            "Type : Type du véhicule", ["A", "B", "C", "D", "E", "F"], key="type1"
        )
        occupation = st.selectbox(
            "Profession du conducteur ",
            ["Employed", "Self-employed", "Housewife", "Unemployed", "Retired"],
            key="occupation1",
        )
        age = st.number_input(
            "Age", min_value=0, max_value=100, value=25, step=1, key="age1"
        )

    with col2:
        bonus = st.number_input(
            "Réduction",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="bonus1",
        )

        exppdays = st.number_input(
            "Exposition au risque (en jours)",
            min_value=None,
            max_value=None,
            value=0.0,
            step=0.01,
            format="%.2f",
            key="exppdays2",
        )
        poldur = st.number_input(
                "Numero de contrat",
                min_value=1900,
                max_value=2100,
                value=2020,
                step=1,
                key="ploNum2",
            )

    group1 = st.number_input(
        "Groupe de la voiture",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        key="group1",
    )

    adind = st.selectbox("Adind", [0, 1], key="adind2")

    density = st.number_input(
        "Densité de la population",
        min_value=None,
        max_value=None,
        value=0.0,
        step=0.01,
        format="%.2f",
        key="density2",
        )
    value = st.number_input(
        "valeur",
        min_value=1,
        max_value=500000,
        value=1000,
        step=1,
        key="value2",
    )


    submit = st.button("Prédire votre prime 💰", key="predict")

    if submit:
        data = {    "Type": Type,
            "Occupation": occupation,
            "Age": age,
            "Group1": group1,
            "Bonus": bonus,
            "Poldur": poldur,
            "Value": value,
            "Adind": adind,
            "Density": density,
            "Exppdays": exppdays}

        data = json.dumps(data)
        response = requests.post(url= 'http://127.0.0.1:8000/predict/', data= data)

        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Résultat de la prédiction : {prediction}")
        else:
            st.error('Erreur dans l\'appel API')




