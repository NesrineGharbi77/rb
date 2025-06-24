#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Streamlit pour traitement de relevés bancaires
OCR → Pré-détection → DeepSeek → Interface modifiable → Export
"""

import streamlit as st
import pandas as pd
import os
import re
import cv2
import json
import traceback
import tempfile
from typing import List, Tuple, Dict
from io import BytesIO

from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from openai import OpenAI

# ───────────────────────── CONFIG ──────────────────────────
X_MIN_PCT = 60.0
AMT_CLUSTER_GAP = 3.0
OCR_LANG = "fr"
API_KEY = st.secrets["DEEPSEEK_API_KEY"]  # Clé API depuis les secrets Streamlit
API_BASE_URL = "https://api.deepseek.com"

# Configuration de la page
st.set_page_config(
    page_title="Traitement Relevés Bancaires",
    page_icon="🏦",
    layout="wide"
)

# ───────────────────────── CACHE OCR ────────────────────────────────
@st.cache_resource
def init_ocr():
    return PaddleOCR(use_angle_cls=True, lang=OCR_LANG)

# ───────────────────────── FONCTIONS CORE (reprises du code original) ──────────────────────────
def extract_text_coords_pct(img_path: str, ocr_engine) -> Tuple[List[Tuple[str, float, float]], int]:
    """Retourne [(texte, x_pct, y_px)] et largeur image."""
    img = cv2.imread(img_path)
    if img is None:
        return [], 1
    h, w = img.shape[:2]
    out = []
    for page in ocr_engine.ocr(img, cls=True):
        for line in page:
            txt, conf = line[1]
            if conf < 0.5:
                continue
            bbox = line[0]
            x_px = min(p[0] for p in bbox)
            y_px = min(p[1] for p in bbox)
            x_pct = x_px / w * 100
            out.append((txt, x_pct, y_px))
    return out, w

def cluster_x(xs: List[float], gap: float = AMT_CLUSTER_GAP) -> List[List[float]]:
    if not xs:
        return []
    xs.sort()
    clusters, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > gap:
            clusters.append(cur)
            cur = [x]
        else:
            cur.append(x)
    clusters.append(cur)
    return clusters

def extract_transactions(page_items: List[Tuple[str, float, float]]) -> List[Dict]:
    """
    Renvoie [{'debit': 123.45, 'credit': None}, …] pour la page
    """
    pattern = re.compile(r'([+-]?)\s*(\d{1,3}(?:[\.\s]\d{3})*(?:,\d{2}))')
    candidates = []
    
    for txt, x_pct, y in page_items:
        if x_pct < X_MIN_PCT:
            continue
        
        match = pattern.search(txt)
        if not match:
            continue
            
        signe = match.group(1).strip() if match.group(1) else ""
        montant_brut = match.group(2)
        
        clean = montant_brut.replace('.', '').replace(' ', '').replace(',', '.')
        try:
            val = float(clean)
        except ValueError:
            continue
            
        candidates.append((val, x_pct, y, signe))

    if not candidates:
        return []

    xs_sans_signe = [x for _, x, _, signe in candidates if not signe]
    clust = cluster_x(xs_sans_signe) if xs_sans_signe else []
    
    if len(clust) >= 2:
        debit_x = sum(clust[0]) / len(clust[0])
        credit_x = sum(clust[-1]) / len(clust[-1])
    else:
        debit_x = credit_x = None

    txs = []
    for val, x_pct, y, signe in sorted(candidates, key=lambda t: (t[2], t[1])):
        tx = {'debit': None, 'credit': None}
        
        if signe == "-":
            tx['debit'] = val
        elif signe == "+":
            tx['credit'] = val
        elif not signe:
            if debit_x and credit_x and abs(x_pct - debit_x) < abs(x_pct - credit_x):
                tx['debit'] = val
            else:
                tx['credit'] = val
        
        txs.append(tx)
    
    return txs

def call_deepseek(text: str, pre_tx: List[Dict]) -> Dict:
    """Appel à l'API DeepSeek"""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    fil = [{k: v for k, v in tx.items() if v is not None} for tx in pre_tx]
    info = ""
    if fil:
        info = "\nPré-détection débit/crédit (par signes et position) :\n" + json.dumps(fil, indent=2, ensure_ascii=False)

    prompt = (
        "Tu es un expert en extraction d'informations depuis des relevés bancaires. "
        "Analyse le texte suivant et retourne le JSON structuré selon le schéma suivant :\n\n"
        "{\n"
        '    "compte": "string|null",\n'
        '    "titulaire": "string|null",\n'
        '    "banque": "string|null",\n'
        '    "date_releve": "JJ/MM/AAAA|null",\n'
        '    "solde_initial": "float|null",\n'
        '    "solde_final": "float|null",\n'
        '    "transactions": [\n'
        '        {\n'
        '            "date comptable": "JJ/MM/AAAA|null",\n'
        '            "date valeur": "JJ/MM/AAAA|null",\n'
        '            "libellé": "string|null",\n'
        '            "débit": "float|null",\n'
        '            "crédit": "float|null"\n'
        '        }\n'
        '    ],\n'
        '    "total_debit": "float",\n'
        '    "total_credit": "float"\n'
        "}\n\n"
        "1. Calcule et remplis « total_debit » et « total_credit » à partir des transactions.\n"
        "2. Vérifie si des montants de débit se trouvent dans le champ crédit (ou inversement).\n"
        "3. Ignorer les tables DETAIL DES PRELEVEMENTS/virements SEPA RECUS\n"
        "4. Ignorer les lignes vides (sans date, libellé ou montant)\n"
        "5. Exclure les soldes et totaux de page.\n\n"
        "Une transaction doit avoir soit un montant au débit soit un montant au crédit, jamais les deux.\n"
        f"{info}\nTexte OCR :\n{text}"
    )
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        return {"error": f"Erreur API : {e}"}

    m = re.search(r'(\{(?:.|\n)*\})', raw)
    if not m:
        return {"error": "Impossible d'extraire un bloc JSON", "raw_response": raw}

    bloc = m.group(1)
    try:
        return json.loads(bloc)
    except json.JSONDecodeError as e:
        return {"error": f"JSONDecodeError: {e}", "extracted_block": bloc}

def process_page(img, ocr_engine) -> Tuple[str, Dict]:
    """Traite une page d'image"""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name, "PNG")
        items, _ = extract_text_coords_pct(tmp.name, ocr_engine)
        os.unlink(tmp.name)
    
    full_text = "\n".join([t for t, _, _ in items])
    pre_tx = extract_transactions(items)
    result = call_deepseek(full_text, pre_tx)
    
    return full_text, result

# ───────────────────────── INTERFACE STREAMLIT ──────────────────────────
def main():
    st.title("🏦 Traitement de Relevés Bancaires")
    st.markdown("Pipeline OCR → IA → Tableau modifiable → Export")
    
    # Vérification de la clé API
    try:
        api_key_test = st.secrets["DEEPSEEK_API_KEY"]
    except Exception as e:
        st.error("❌ Clé API DeepSeek non configurée dans les secrets Streamlit")
        st.stop()
    
    # Zone d'upload
    st.header("📁 Upload du PDF")
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre relevé bancaire (PDF)",
        type=['pdf'],
        help="Sélectionnez un fichier PDF de relevé bancaire"
    )
    
    if uploaded_file is None:
        st.info("👆 Uploadez un fichier PDF pour commencer")
        return
    
    # Initialisation OCR
    with st.spinner("Initialisation de l'OCR..."):
        ocr_engine = init_ocr()
    
    # Conversion PDF en images
    with st.spinner("Conversion du PDF..."):
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            images = convert_from_path(tmp_path, dpi=350)
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Erreur lors de la conversion PDF : {e}")
            return
    
    st.success(f"✅ PDF converti - {len(images)} page(s) détectée(s)")
    
    # Sélection des pages
    st.header("📄 Sélection des pages")
    
    if len(images) == 1:
        selected_pages = [0]
        st.info("Une seule page détectée - sélection automatique")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            select_all = st.checkbox("Sélectionner toutes les pages")
            if select_all:
                selected_pages = list(range(len(images)))
            else:
                selected_pages = st.multiselect(
                    "Choisissez les pages à traiter",
                    options=list(range(len(images))),
                    format_func=lambda x: f"Page {x+1}",
                    default=[0]
                )
        
        with col2:
            if selected_pages:
                # Aperçu des pages sélectionnées
                preview_cols = st.columns(min(3, len(selected_pages)))
                for i, page_idx in enumerate(selected_pages[:3]):
                    with preview_cols[i]:
                        st.image(images[page_idx], caption=f"Page {page_idx+1}", use_column_width=True)
    
    if not selected_pages:
        st.warning("Veuillez sélectionner au moins une page")
        return
    
    # Initialisation des données dans le state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    
    # Traitement des pages
    st.header("🔄 Traitement et Résultats")
    
    # Bouton de traitement
    if st.button("🚀 Lancer le traitement", type="primary"):
        progress_bar = st.progress(0)
        
        for i, page_idx in enumerate(selected_pages):
            with st.spinner(f"Traitement page {page_idx+1}..."):
                try:
                    full_text, result = process_page(images[page_idx], ocr_engine)
                    st.session_state.processed_data[page_idx] = {
                        'result': result,
                        'full_text': full_text,
                        'image': images[page_idx]
                    }
                except Exception as e:
                    st.error(f"Erreur page {page_idx+1}: {e}")
                    st.session_state.processed_data[page_idx] = {
                        'result': {'error': str(e)},
                        'full_text': '',
                        'image': images[page_idx]
                    }
            
            progress_bar.progress((i + 1) / len(selected_pages))
        
        st.success("✅ Traitement terminé !")
    
    # Affichage des résultats par page
    if st.session_state.processed_data:
        
        # Sélecteur de page pour affichage
        st.subheader("📊 Résultats par page")
        display_page = st.selectbox(
            "Page à afficher/modifier",
            options=[p for p in selected_pages if p in st.session_state.processed_data],
            format_func=lambda x: f"Page {x+1}"
        )
        
        if display_page in st.session_state.processed_data:
            page_data = st.session_state.processed_data[display_page]
            result = page_data['result']
            
            # Layout en colonnes
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(f"📄 Page {display_page + 1}")
                
                # Aperçu de l'image
                st.image(page_data['image'], caption=f"Page {display_page + 1}", use_column_width=True)
                
                # Informations générales
                if 'error' not in result:
                    st.subheader("ℹ️ Informations générales")
                    info_data = {
                        'Compte': result.get('compte', 'N/A'),
                        'Titulaire': result.get('titulaire', 'N/A'),
                        'Banque': result.get('banque', 'N/A'),
                        'Date relevé': result.get('date_releve', 'N/A'),
                        'Solde initial': result.get('solde_initial', 'N/A'),
                        'Solde final': result.get('solde_final', 'N/A'),
                    }
                    
                    for key, value in info_data.items():
                        st.text(f"{key}: {value}")
                else:
                    st.error(f"Erreur: {result['error']}")
            
            with col2:
                st.subheader("✏️ Transactions - Tableau modifiable")
                
                if 'error' not in result and 'transactions' in result:
                    transactions = result['transactions']
                    
                    if transactions:
                        # Conversion en DataFrame
                        df = pd.DataFrame(transactions)
                        
                        # Éditeur de données
                        edited_df = st.data_editor(
                            df,
                            num_rows="dynamic",
                            use_container_width=True,
                            key=f"editor_{display_page}"
                        )
                        
                        # Calcul des totaux
                        total_debit = edited_df['débit'].fillna(0).sum()
                        total_credit = edited_df['crédit'].fillna(0).sum()
                        
                        col_tot1, col_tot2 = st.columns(2)
                        with col_tot1:
                            st.metric("Total Débits", f"{total_debit:.2f} €")
                        with col_tot2:
                            st.metric("Total Crédits", f"{total_credit:.2f} €")
                        
                        # Mise à jour des données modifiées
                        if st.button(f"💾 Sauvegarder les modifications - Page {display_page + 1}"):
                            # Mise à jour du result avec les données modifiées
                            updated_transactions = edited_df.to_dict('records')
                            st.session_state.processed_data[display_page]['result']['transactions'] = updated_transactions
                            st.session_state.processed_data[display_page]['result']['total_debit'] = total_debit
                            st.session_state.processed_data[display_page]['result']['total_credit'] = total_credit
                            st.success("✅ Modifications sauvegardées !")
                    else:
                        st.info("Aucune transaction détectée sur cette page")
                else:
                    st.warning("Impossible d'afficher les transactions (erreur de traitement)")
        
        # Section Export
        st.header("📤 Export des données")
        
        # Consolidation des données de toutes les pages
        all_transactions = []
        all_info = {}
        
        for page_idx, page_data in st.session_state.processed_data.items():
            result = page_data['result']
            if 'error' not in result and 'transactions' in result:
                # Ajout du numéro de page à chaque transaction
                for tx in result['transactions']:
                    tx_copy = tx.copy()
                    tx_copy['page'] = page_idx + 1
                    all_transactions.append(tx_copy)
                
                # Mise à jour des infos générales (prend la première page valide)
                if not all_info:
                    all_info = {k: v for k, v in result.items() if k != 'transactions'}
        
        if all_transactions:
            export_df = pd.DataFrame(all_transactions)
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Export Excel
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    export_df.to_excel(writer, sheet_name='Transactions', index=False)
                    
                    # Feuille avec informations générales
                    info_df = pd.DataFrame([all_info])
                    info_df.to_excel(writer, sheet_name='Informations', index=False)
                
                st.download_button(
                    label="📊 Télécharger Excel",
                    data=buffer.getvalue(),
                    file_name=f"releve_bancaire_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col_exp2:
                # Export CSV
                csv_buffer = BytesIO()
                export_df.to_csv(csv_buffer, index=False, encoding='utf-8')
                
                st.download_button(
                    label="📄 Télécharger CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"releve_bancaire_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv"
                )
            
            # Aperçu des données consolidées
            st.subheader("👀 Aperçu des données consolidées")
            st.dataframe(export_df, use_container_width=True)
            
            # Statistiques finales
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Total transactions", len(export_df))
            with col_stat2:
                st.metric("Total débits", f"{export_df['débit'].fillna(0).sum():.2f} €")
            with col_stat3:
                st.metric("Total crédits", f"{export_df['crédit'].fillna(0).sum():.2f} €")
        else:
            st.warning("Aucune donnée à exporter")

if __name__ == "__main__":
    main()
