#!/bin/bash

# Script pour lancer le dashboard Streamlit du modèle de stock picking

echo "Lancement du dashboard Stock Picking Model..."

# Vérifier si Streamlit est installé
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit n'est pas installé. Installation en cours..."
    pip install streamlit
fi

# Lancer le dashboard
streamlit run dashboard.py
