import streamlit as st
import pandas as pd
import joblib

# Chargement des fichiers nécessaires
model = joblib.load("C:/Users/LENOVO/Documents/Model+Code/random_forest_model.joblib")
preprocessor = joblib.load("C:/Users/LENOVO/Documents/Model+Code/processor.joblib")

# Configuration de la page
st.set_page_config(page_title="App Prédiction Risque Crédit", layout="wide", page_icon="💳")
st.title("💳 Prédiction du Risque de Défaut de Crédit")
st.markdown("Ce modèle prédit si un client présente un **risque de défaut de crédit** basé sur des caractéristiques financières et comportementales.")

# Création des onglets
tabs = st.tabs(["📊 Prédiction", "ℹ️ Description des Variables"])

# Onglet Prédiction
with tabs[0]:
    st.header("Remplissez les informations client")

    col1, col2 = st.columns(2)

    with col1:
        transactions_suspectes = st.slider("Nombre de transactions suspectes", 0, 100, 2)
        localisation_inhabituelle = st.selectbox("Localisation inhabituelle", [0, 1])
        heure_transaction = st.selectbox("Heure de transaction", ["Jour", "Nuit"])
        anciennete_emploi = st.slider("Ancienneté en emploi (mois)", 0, 480, 60)
        taux_endettement = st.slider("Taux d'endettement (%)", 0.0, 100.0, 30.0)
        defaut_credit = st.selectbox("A déjà eu un défaut de crédit ?", [0, 1])

    with col2:
        montant_credit = st.number_input("Montant du crédit demandé", min_value=0.0, value=10000.0)
        revenu_mensuel = st.number_input("Revenu mensuel", min_value=0.0, value=2000.0)
        age = st.slider("Âge", 18, 75, 35)
        nombre_credits_en_cours = st.slider("Nombre de crédits en cours", 0, 10, 1)
        type_contrat = st.selectbox("Type de contrat", ["CDI", "CDD", "Indépendant", "Sans emploi"])
        situation_matrimoniale = st.selectbox("Situation matrimoniale", ["Célibataire", "Marié(e)", "Divorcé(e)"])
        sexe = st.selectbox("Sexe", ["Homme", "Femme"])
        historique_remboursement = st.selectbox("Historique de remboursement", ["Bon", "Moyen", "Mauvais"])

    # Dictionnaire d'entrée
    input_dict = {
        "transactions_suspectes": transactions_suspectes,
        "localisation_inhabituelle": localisation_inhabituelle,
        "heure_transaction": heure_transaction,
        "anciennete_emploi": anciennete_emploi,
        "taux_endettement": taux_endettement,
        "montant_credit": montant_credit,
        "revenu_mensuel": revenu_mensuel,
        "age": age,
        "nombre_credits_en_cours": nombre_credits_en_cours,
        "type_contrat": type_contrat,
        "situation_matrimoniale": situation_matrimoniale,
        "sexe": sexe,
        "historique_remboursement": historique_remboursement,
        "defaut_credit": defaut_credit,
    }

    seuil = st.slider("📊 Seuil de détection du risque", 0.0, 1.0, 0.4, 0.05)

    if st.button("🔍 Prédire le risque"):
        try:
            # Conversion en DataFrame
            input_data = pd.DataFrame([input_dict])

            # Prétraitement
            input_prepared = preprocessor.transform(input_data)

            # Prédiction
            proba = model.predict_proba(input_prepared)[0][1]
            prediction = int(proba > seuil)

            st.write(f"🧪 Probabilité brute de risque : {proba:.4f}")
            if prediction == 1:
                st.error(f"❌ Risque de défaut détecté avec une probabilité de {proba:.2%}")
            else:
                st.success(f"✅ Pas de risque détecté. Probabilité de défaut : {proba:.2%}")
        except Exception as e:
            st.warning(f"⚠️ Erreur lors de la prédiction : {e}")


# Onglet Description
with tabs[1]:
    st.header("📘 Description des variables utilisées")
    st.markdown("""
    | Variable                          | Description |
    |-----------------------------------|-------------|
    | **transactions_suspectes**        | Nombre de transactions jugées anormales |
    | **localisation_inhabituelle**     | Si le client effectue des transactions depuis des lieux inhabituels |
    | **heure_transaction**             | Plage horaire des opérations (Jour/Nuit) |
    | **anciennete_emploi**             | Nombre de mois depuis l'embauche actuelle |
    | **taux_endettement**              | Ratio entre les dettes et les revenus (%) |
    | **montant_credit**                | Montant du crédit demandé |
    | **revenu_mensuel**                | Revenu net mensuel du client |
    | **age**                           | Âge du client |
    | **nombre_credits_en_cours**       | Nombre de crédits en cours actuellement |
    | **type_contrat**                  | Type de contrat professionnel (CDI, CDD, etc.) |
    | **situation_matrimoniale**        | Statut matrimonial (Célibataire, Marié, etc.) |
    | **sexe**                          | Sexe du client |
    | **historique_remboursement**      | Historique de remboursement (Bon, Moyen, Mauvais) |
    | **defaut_credit**                 | Si le client a déjà eu un défaut de crédit par le passé (Oui = 1, Non = 0) |
    | **risque_fraude** *(cible)*       | Indique si le client présente un **risque élevé de fraude ou de défaut** de crédit |
    """)

