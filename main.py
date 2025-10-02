import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import warnings
from typing import Optional
import re
import os
import base64
from src.helpers import detect_tension_level, parse_time_ranges, get_time_periods, get_segment, calculate_business_hours, normalize_text, normalize_json_object
from src.turpe_calculator import TURPECalculator

# Import du nouveau module des primes de risque
from src.risk_premiums import RiskPremiumCalculator, CTACalculator
import json
import io
import zipfile
import holidays

warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Pricing Électricité BtB",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FONCTIONS D'IMPORT AMÉLIORÉES (basées sur data_loader.py)
# ============================================================================

def load_enedis_perimeter(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge et valide un fichier de périmètre ENEDIS avec gestion robuste des formats
    """
    try:
        # Tentative de lecture avec différents encodages
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        
        # Standardisation des noms de colonnes
        original_columns = df.columns
        df.columns = df.columns.str.lower().str.strip()
        
        # Mapping des colonnes courantes ENEDIS
        column_mapping = {
            'point_reference_mesure': 'prm',
            'prm_id': 'prm',
            'identifiant prm': 'prm',
            'identifiant_prm': 'prm',
            'lat': 'latitude',
            'lng': 'longitude',
            'lon': 'longitude',
            'code_naf_ape': 'code_naf',
            'naf': 'code_naf',
            'raison_sociale': 'company_name',
            'nom': 'company_name',
            'adresse': 'address',
            'ville': 'city',
            'code_postal': 'postal_code',
            'puissance_souscrite': 'subscribed_power',
            'puissance limite soutirage': 'Puissance limite soutirage', # Keep original case for helpers
            'domaine de tension': 'Domaine de tension', # Keep original case for helpers
            'tarif': 'tariff'
        }
        
        column_mapping['formule tarifaire acheminement'] = 'Formule tarifaire acheminement'

        # Itérer sur les colonnes originales pour mapper les postes horaires et classes temporelles
        for col in original_columns:
            # Garder le nom original pour les colonnes structurées nécessaires au calcul
            if re.match(r'poste horaire \d', col, re.IGNORECASE) or \
               re.match(r'classe temporelle \d', col, re.IGNORECASE):
                column_mapping[col.lower().strip()] = col

        # Application du mapping
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Validation des colonnes obligatoires
        required_columns = ['prm']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes dans le fichier périmètre : {missing_columns}")
            st.error("Colonnes détectées dans le fichier :")
            st.write(list(df.columns))
            return None
        
        # Nettoyage des données
        df = df.dropna(subset=['prm'])
        df['prm'] = df['prm'].astype(str).str.strip()
        
        # Conversion des coordonnées GPS si présentes
        if 'latitude' in df.columns:
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        if 'longitude' in df.columns:
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Ajout de colonnes par défaut si manquantes
        default_columns = {
            'code_naf': '0000Z',
            'company_name': 'Non renseigné',
            'address': 'Non renseigné',
            'city': 'Non renseigné',
            'postal_code': '00000'
        }
        
        for col, default_value in default_columns.items():
            if col not in df.columns:
                df[col] = default_value

        # S'assurer que les noms de société vides sont bien remplacés
        if 'company_name' in df.columns:
            df['company_name'] = df['company_name'].fillna('Non renseigné')

        # --- NOUVELLES FONCTIONNALITÉS ---
        # Déterminer le domaine de tension et les plages horaires
        st.info("Analyse des caractéristiques des sites (tension, plages horaires, segment)...")
        df['tension_level'] = df.apply(detect_tension_level, axis=1)
        df['time_ranges'] = df.apply(parse_time_ranges, axis=1)

        # NOUVEAU: Ajout du segment TURPE
        if 'Formule tarifaire acheminement' not in df.columns:
            df['Formule tarifaire acheminement'] = 'N/A'
        df['segment'] = df.apply(lambda row: get_segment(row['tension_level'], row['Formule tarifaire acheminement']), axis=1)

        st.success(f"Fichier périmètre chargé et enrichi avec succès : {len(df)} sites")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier périmètre : {str(e)}")
        st.exception(e)
        return None


def load_enedis_consumption(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge et valide un fichier de consommation ENEDIS avec gestion robuste des formats
    """
    try:
        # Tentative de lecture avec différents encodages
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        except Exception:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
        
        # Standardisation des noms de colonnes
        df.columns = df.columns.str.lower().str.strip()
        
        # Mapping des colonnes courantes ENEDIS
        column_mapping = {
            'point_reference_mesure': 'prm',
            'prm_id': 'prm',
            'identifiant prm': 'prm',
            'identifiant_prm': 'prm',
            'horodate': 'datetime',
            'date_heure': 'datetime',
            'timestamp': 'datetime',
            'date': 'datetime',
            'valeur': 'power_value',  # Changé pour être plus générique
            'consommation': 'power_value',
            'consumption': 'power_value',
            'energy': 'power_value',
            'kwh': 'power_value',
            'pas': 'time_step',
            'step': 'time_step',
            'intervalle': 'time_step',
            'unité': 'unit',
            'unite': 'unit',
            'unit': 'unit'
        }
        
        # Application du mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Validation des colonnes obligatoires
        required_columns = ['prm', 'datetime', 'power_value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes dans le fichier consommation : {missing_columns}")
            st.error("Colonnes détectées dans le fichier :")
            st.write(list(df.columns))
            return None
        
        # NOUVEAU: Filtrage par unité (uniquement W, kW, MW)
        if 'unit' in df.columns:
            valid_units = ['W', 'kW', 'MW']
            initial_count = len(df)
            df = df[df['unit'].isin(valid_units)]
            filtered_count = len(df)
            if filtered_count < initial_count:
                st.info(f"Filtré par unité : {initial_count - filtered_count} lignes supprimées (unités non valides)")
            
            # Conversion en watts pour homogénéiser
            df['power_w'] = df.apply(lambda row: 
                row['power_value'] if row['unit'] == 'W'
                else row['power_value'] * 1000 if row['unit'] == 'kW'
                else row['power_value'] * 1000000 if row['unit'] == 'MW'
                else row['power_value'], axis=1)
        else:
            # Si pas d'unité, assumer que c'est en W
            df['power_w'] = df['power_value']
            st.warning("Pas de colonne 'Unité' trouvée, valeurs assumées en Watts")
        
        # Nettoyage et conversion des données
        df = df.dropna(subset=['prm', 'datetime', 'power_w'])
        df['prm'] = df['prm'].astype(str).str.strip()
        
        # Conversion de la colonne datetime avec formats multiples
        try:
            datetime_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%d/%m/%Y %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%d/%m/%Y %H:%M',
                '%Y-%m-%d',
                '%d/%m/%Y'
            ]
            
            datetime_converted = False
            for fmt in datetime_formats:
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'], format=fmt)
                    datetime_converted = True
                    st.info(f"Format datetime détecté : {fmt}")
                    break
                except:
                    continue
            
            if not datetime_converted:
                df['datetime'] = pd.to_datetime(df['datetime'], infer_datetime_format=True)
                st.info("Format datetime détecté automatiquement")
                
        except Exception as e:
            st.error(f"Impossible de convertir la colonne datetime : {str(e)}")
            return None
        
        # Conversion des consommations en numérique
        df['power_w'] = pd.to_numeric(df['power_w'], errors='coerce')
        df = df.dropna(subset=['power_w'])
        
        # Validation des valeurs de consommation
        negative_values = (df['power_w'] < 0).sum()
        if negative_values > 0:
            st.warning(f"Attention : {negative_values} valeurs de puissance négatives détectées")
        
        # Tri par datetime
        df = df.sort_values(['prm', 'datetime'])
        
        # NOUVEAU: Détection du pas de temps à partir du champ "Pas" si disponible
        detected_step_str = "Non détecté"
        if 'time_step' in df.columns and not df['time_step'].isna().all():
            # Parser le format ISO 8601 (PT5M, PT15M, PT1H, etc.)
            step_value = df['time_step'].iloc[0]
            if isinstance(step_value, str) and step_value.startswith('PT'):
                try:
                    if step_value.endswith('M'):
                        minutes = int(step_value[2:-1])
                        detected_step_str = f"{minutes} minutes"
                    elif step_value.endswith('H'):
                        hours = int(step_value[2:-1])
                        detected_step_str = f"{hours} heure(s)"
                except:
                    pass
        
        # Fallback: détection par différence de timestamps
        if detected_step_str == "Non détecté" and len(df) > 1:
            time_diff = df['datetime'].diff().mode()
            if not time_diff.empty:
                time_step = time_diff[0]
                if time_step == pd.Timedelta(hours=1):
                    detected_step_str = "1 heure"
                elif time_step == pd.Timedelta(minutes=30):
                    detected_step_str = "30 minutes"
                elif time_step == pd.Timedelta(minutes=15):
                    detected_step_str = "15 minutes"
                elif time_step == pd.Timedelta(minutes=5):
                    detected_step_str = "5 minutes"
                else:
                    detected_step_str = f"{time_step}"
        
        st.info(f"Pas de temps détecté : {detected_step_str}")
        
        st.success(f"Fichier consommation chargé avec succès : {len(df)} mesures")
        
        # Statistiques sur les données
        date_min = df['datetime'].min()
        date_max = df['datetime'].max()
        power_mean = df['power_w'].mean()
        power_max = df['power_w'].max()
        
        st.info(f"Période : {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")
        st.info(f"Puissance moyenne : {power_mean:.2f} W, Maximum : {power_max:.2f} W")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier consommation : {str(e)}")
        return None


def validate_data_consistency(perimeter_df: pd.DataFrame, consumption_data: dict) -> dict:
    """
    Valide la cohérence entre les données de périmètre et de consommation
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        perimeter_prms = set(perimeter_df['prm'].astype(str))
        consumption_prms = set(consumption_data.keys())
        
        # PRMs dans le périmètre mais sans données de consommation
        missing_consumption = perimeter_prms - consumption_prms
        if missing_consumption:
            validation_results['warnings'].append(
                f"{len(missing_consumption)} PRMs sans données de consommation"
            )
        
        # PRMs avec consommation mais pas dans le périmètre
        missing_perimeter = consumption_prms - perimeter_prms
        if missing_perimeter:
            validation_results['warnings'].append(
                f"{len(missing_perimeter)} PRMs avec consommation mais absents du périmètre"
            )
        
        # PRMs communs
        common_prms = perimeter_prms & consumption_prms
        validation_results['stats'] = {
            'perimeter_sites': len(perimeter_prms),
            'consumption_sites': len(consumption_prms),
            'common_sites': len(common_prms),
            'missing_consumption': len(missing_consumption),
            'missing_perimeter': len(missing_perimeter)
        }
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Erreur lors de la validation : {str(e)}")
    
    return validation_results

# ============================================================================
# INITIALISATION ET FONCTIONS UTILITAIRES
# ============================================================================

@st.cache_data
def load_naf_codes(filepath="data/naf_codes.xlsx"):
    """Charge le fichier des codes NAF."""
    try:
        df = pd.read_excel(filepath)
        # Créer une colonne pour l'affichage dans le selectbox
        df['display'] = df['Code NAF'] + " - " + df['Activité']
        return df
    except FileNotFoundError:
        st.error(f"Le fichier des codes NAF est introuvable à l'emplacement : {filepath}")
        return pd.DataFrame(columns=['Code NAF', 'Activité', 'Eligible CEE', 'display'])

def init_session_state():
    """Initialise les variables de session si elles n'existent pas"""
    # Données Client
    if 'client_info' not in st.session_state:
        st.session_state.client_info = {
            'company_name': '', 'siren': '', 'naf_code': None,
            'naf_activity': None, 'cee_eligible': None
        }

    # Données de la Cotation
    if 'quotation_info' not in st.session_state:
        st.session_state.quotation_info = {
            'quotation_id': None, 'quotation_date': datetime.now().date(),
            'quotation_time': datetime.now().time(), 'start_date': date.today(),
            'end_date': date.today() + timedelta(days=365),
            'offer_type': "Prix fixe unique", 'pricing_type': "Prix lissé",
            'status': "Indicatif",
            'validity_end_date': datetime.now().date(),
            'validity_end_time': datetime.now().time(),
            'validity_duration_hours': 0.0
        }

    # Le reste de l'état de session
    if 'perimeter_df' not in st.session_state: st.session_state.perimeter_df = None
    if 'consumption_data' not in st.session_state: st.session_state.consumption_data = {}
    if 'price_curve_df' not in st.session_state: st.session_state.price_curve_df = None
    if 'weather_data' not in st.session_state: st.session_state.weather_data = {}
    if 'forecast_results' not in st.session_state: st.session_state.forecast_results = {}
    if 'pricing_results' not in st.session_state: st.session_state.pricing_results = {}
    if 'premiums_df' not in st.session_state: st.session_state.premiums_df = None
    if 'site_details' not in st.session_state: st.session_state.site_details = {}

    # Configuration des primes de risque
    if 'risk_premiums_config' not in st.session_state:
        st.session_state.risk_premiums_config = {
            'default_premiums': {
                "Délai de validité": 1.0,
                "Coût d'équilibrage": 2.0,
                "Coût du profil": 1.0,
                "Risque volume": 2.0,
                "Fluctuation du parc": 0.0,
                "Recalage Prix Live": 0.0,
                "Délai de Paiement": 0.29,
                "Risque Crédit": 0.5
            },
            'validity_params': {}
        }

    if 'app_tariffs' not in st.session_state:
        try:
            with open('tariffs_config.json', 'r') as f:
                st.session_state.app_tariffs = json.load(f)
            st.toast("Tarifs chargés depuis `tariffs_config.json`.")
        except (FileNotFoundError, json.JSONDecodeError):
            st.toast("Fichier `tariffs_config.json` non trouvé ou invalide, utilisation des tarifs par défaut.")
            st.session_state.app_tariffs = {} # Start with empty dict

        # --- Définition de la structure de tarifs par défaut ---
        default_tariffs = {
            "Marge": {"marge_commerciale": 5.0},
            "Autres_Couts": {
                "CEE": 12.0,
                "Garantie_Capacite": 3.0,
                "Garanties_Origine": 0.8
            },
            "Primes_Risques": {
                "Délai de validité": 1.0,
                "Coût d'équilibrage": 2.0,
                "Coût du profil": 1.0,
                "Risque volume": 2.0,
                "Fluctuation du parc": 0.0,
                "Recalage Prix Live": 0.0,
                "Délai de Paiement": 0.29,
                "Risque Crédit": 0.5
            },
            "Taxes": {
                "cta_rate": 0.2193, "excise_tax": 25.79,
                "vat_rate": 0.20, "c3s_rate": 0.0016
            },
            "TURPE": {
                "CG_HTA": {"CARD": 499.80, "UNIQUE": 435.72, "INJECTION": 717.60},
                "CG_BT_SUP36": {"CARD": 249.84, "UNIQUE": 217.80, "INJECTION": 358.80},
                "CG_BT_INF36": {"CARD": 18.00, "UNIQUE": 16.80, "INJECTION": 26.40},
                "CC_HTA": 376.39, "CC_BT_SUP36": 283.27, "CC_BT_INF36": 22.00,
                "HTA_PUISSANCE": {
                    "CU_FIXE": {"POINTE": 14.41, "HPH": 14.41, "HCH": 14.41, "HPB": 12.55, "HCB": 11.22},
                    "LU_FIXE": {"POINTE": 35.33, "HPH": 32.30, "HCH": 20.39, "HPB": 14.33, "HCB": 11.56}
                },
                "HTA_ENERGIE": {
                    "CU_FIXE": {"POINTE": 5.74, "HPH": 4.23, "HCH": 1.99, "HPB": 1.01, "HCB": 0.69},
                    "LU_FIXE": {"POINTE": 2.65, "HPH": 2.10, "HCH": 1.47, "HPB": 0.92, "HCB": 0.68}
                },
                "BT_SUP36_PUISSANCE": {
                    "CU": {"HPH": 17.61, "HCH": 15.96, "HPB": 14.56, "HCB": 11.98},
                    "LU": {"HPH": 30.16, "HCH": 21.18, "HPB": 16.64, "HCB": 12.37}
                },
                "BT_SUP36_ENERGIE": {
                    "CU": {"HPH": 6.91, "HCH": 4.21, "HPB": 2.13, "HCB": 1.52},
                    "LU": {"HPH": 5.69, "HCH": 3.47, "HPB": 2.01, "HCB": 1.49}
                },
                "CMDPS_HTA_COEFF": 0.04, "CMDPS_BT_SUP36": 12.41
            }
        }

        # --- Fusionner les tarifs chargés avec les défauts pour assurer la compatibilité ---
        for key, default_value in default_tariffs.items():
            if key not in st.session_state.app_tariffs:
                st.session_state.app_tariffs[key] = default_value
            elif isinstance(default_value, dict):
                # Fusionner les sous-dictionnaires
                for sub_key, sub_default_value in default_value.items():
                    if sub_key not in st.session_state.app_tariffs[key]:
                        st.session_state.app_tariffs[key][sub_key] = sub_default_value

    # Paramètres tarifaires de la cotation (initialisés à partir des défauts)
    if 'tariff_params' not in st.session_state:
        st.session_state.tariff_params = {
            'marge_commerciale': st.session_state.app_tariffs.get('Marge', {}).get('marge_commerciale', 5.0),
            'cee_costs_yearly': {},
            'capacity_costs_yearly': {},
            'go_costs_yearly': {},
            'green_option_activated': False,
        }

    # Charger les codes NAF
    if 'naf_codes' not in st.session_state:
        st.session_state.naf_codes = load_naf_codes()

# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

def main():
    init_session_state()
    
    # Espace pour le logo du client
    if os.path.exists("assets/client_logo.png"):
        st.image("assets/client_logo.png", width=200)

    st.title("⚡ Pricing d'Offre d'Électricité BtB")
    st.markdown("---")
    
    # Navigation
    tabs = st.tabs([
        "🏢 Paramètres Cotation",
        "📊 Données Sites",
        "✍️ Détail par Site",
        "💰 Courbe de Prix",
        "🔮 Prévision Consommation",
        "⚠️ Paramètres Tarifaires",
        "📋 Résultats Pricing",
        "📤 Export",
        "📚 Référentiel"
    ])
    
    with tabs[0]:
        quotation_parameters_tab()
    
    with tabs[1]:
        sites_data_tab()
    
    with tabs[2]:
        site_details_tab()

    with tabs[3]:
        price_curve_tab()
    
    with tabs[4]:
        forecast_tab()
    
    with tabs[5]:
        risk_premiums_tab()
    
    with tabs[6]:
        pricing_results_tab()
    
    with tabs[7]:
        export_tab()

    with tabs[8]:
        repository_tab()

    # --- Pied de page (version simplifiée) ---
    logo_html = ""
    # On vérifie les deux extensions possibles
    favicon_path_ico = "assets/favicon.ico"
    favicon_path_png = "assets/favicon.png"

    favicon_path = None
    if os.path.exists(favicon_path_ico):
        favicon_path = favicon_path_ico
    elif os.path.exists(favicon_path_png):
        favicon_path = favicon_path_png

    if favicon_path:
        # Déterminer le type MIME en fonction de l'extension du fichier
        mime_type = "image/png" if favicon_path.endswith(".png") else "image/x-icon"

        with open(favicon_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        logo_html = f'<img src="data:{mime_type};base64,{data}" alt="logo" style="height: 24px; margin-right: 15px; vertical-align: middle;">'

    footer_html = f"""
    <hr>
    <div style="text-align: center; padding: 10px;">
        {logo_html}
        <span style="vertical-align: middle;">
            Pricer développé par
            <span style="color: #0044aa; font-weight: bold;">COPILOTE</span>
            <span style="color: #7ed957; font-weight: bold;">ENERGIE</span>
        </span>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def repository_tab():
    """Onglet pour la gestion des référentiels de tarifs et taxes."""
    st.header("📚 Référentiel des Tarifs et Coûts")

    # Utiliser st.form pour regrouper tous les champs et le bouton de sauvegarde
    with st.form(key="repository_form"):
        app_tariffs = st.session_state.app_tariffs

        # --- Marge Commerciale ---
        st.subheader("Marge Commerciale")
        app_tariffs['Marge']['marge_commerciale'] = st.number_input(
            "Marge commerciale par défaut (€/MWh)",
            value=float(app_tariffs.get('Marge', {}).get('marge_commerciale', 5.0)),
            min_value=0.0, step=0.1, format="%.2f"
        )
        st.markdown("---")

        # --- CEE, GC et GO ---
        st.subheader("Autres Coûts par Défaut (€/MWh)")
        autres_couts = app_tariffs.get('Autres_Couts', {})
        col_ac1, col_ac2, col_ac3 = st.columns(3)
        with col_ac1:
            autres_couts['CEE'] = st.number_input("Coût CEE", value=float(autres_couts.get('CEE', 12.0)), min_value=0.0, step=0.1, format="%.2f")
        with col_ac2:
            autres_couts['Garantie_Capacite'] = st.number_input("Garantie de Capacité", value=float(autres_couts.get('Garantie_Capacite', 3.0)), min_value=0.0, step=0.1, format="%.2f")
        with col_ac3:
            autres_couts['Garanties_Origine'] = st.number_input("Garanties d'Origine", value=float(autres_couts.get('Garanties_Origine', 0.8)), min_value=0.0, step=0.1, format="%.2f")
        app_tariffs['Autres_Couts'] = autres_couts
        st.markdown("---")

        # --- Primes de Risques ---
        st.subheader("Primes de Risques par Défaut (€/MWh)")
        primes_risques = app_tariffs.get('Primes_Risques', {})
        pr_col1, pr_col2 = st.columns(2)
        with pr_col1:
            primes_risques["Délai de validité"] = st.number_input("Prime: Délai de validité", value=float(primes_risques.get("Délai de validité", 1.0)), min_value=0.0, step=0.1, format="%.2f")
            primes_risques["Coût d'équilibrage"] = st.number_input("Prime: Coût d'équilibrage", value=float(primes_risques.get("Coût d'équilibrage", 2.0)), min_value=0.0, step=0.1, format="%.2f")
            primes_risques["Coût du profil"] = st.number_input("Prime: Coût du profil", value=float(primes_risques.get("Coût du profil", 1.0)), min_value=0.0, step=0.1, format="%.2f")
            primes_risques["Risque volume"] = st.number_input("Prime: Risque volume", value=float(primes_risques.get("Risque volume", 2.0)), min_value=0.0, step=0.1, format="%.2f")
        with pr_col2:
            primes_risques["Fluctuation du parc"] = st.number_input("Prime: Fluctuation du parc", value=float(primes_risques.get("Fluctuation du parc", 0.0)), min_value=0.0, step=0.1, format="%.2f")
            primes_risques["Recalage Prix Live"] = st.number_input("Prime: Recalage Prix Live", value=float(primes_risques.get("Recalage Prix Live", 0.0)), min_value=0.0, step=0.1, format="%.2f")
            primes_risques["Délai de Paiement"] = st.number_input("Prime: Délai de Paiement", value=float(primes_risques.get("Délai de Paiement", 0.29)), min_value=0.0, step=0.01, format="%.2f")
            primes_risques["Risque Crédit"] = st.number_input("Prime: Risque Crédit", value=float(primes_risques.get("Risque Crédit", 0.5)), min_value=0.0, step=0.1, format="%.2f")
        app_tariffs['Primes_Risques'] = primes_risques
        st.markdown("---")

        # --- Taxes et Contributions ---
        st.subheader("Taxes et Contributions")
        taxes = app_tariffs.get('Taxes', {})
        col_tarif1, col_tarif2, col_tarif3 = st.columns(3)
        with col_tarif1:
            taxes['cta_rate'] = st.number_input(
                "Taux CTA (%)",
                value=float(taxes.get('cta_rate', 0.2193) * 100),
                min_value=0.0, max_value=100.0, step=0.01, format="%.2f",
                help="Contribution Tarifaire d'Acheminement"
            ) / 100
        with col_tarif2:
            taxes['excise_tax'] = st.number_input(
                "Accise électricité (€/MWh)",
                value=float(taxes.get('excise_tax', 22.5)),
                min_value=0.0, step=0.1
            )
        with col_tarif3:
            taxes['vat_rate'] = st.number_input(
                "Taux TVA (%)",
                value=float(taxes.get('vat_rate', 0.20) * 100),
                min_value=0.0, max_value=100.0, step=0.1
            ) / 100
        taxes['c3s_rate'] = st.number_input(
            "Taux C3S (%)",
            value=float(taxes.get('c3s_rate', 0.0016) * 100),
            min_value=0.0, max_value=10.0, step=0.01, format="%.3f",
            help="Contribution Sociale de Solidarité des Sociétés. S'applique sur le chiffre d'affaires hors TVA."
        ) / 100
        app_tariffs['Taxes'] = taxes
        st.info(f"La CTA ({app_tariffs['Taxes']['cta_rate']*100:.2f}%) s'applique sur les composantes fixes du TURPE.")
        st.markdown("---")

        # --- Tarifs TURPE ---
        st.subheader("Tarifs TURPE")
        turpe_tariffs = app_tariffs.get('TURPE', {})
        with st.expander("Composantes de Gestion (€/an)"):
            turpe_tariffs['CG_HTA']['UNIQUE'] = st.number_input("CG HTA - UNIQUE", value=turpe_tariffs['CG_HTA']['UNIQUE'])
            turpe_tariffs['CG_BT_SUP36']['UNIQUE'] = st.number_input("CG BT > 36kVA - UNIQUE", value=turpe_tariffs['CG_BT_SUP36']['UNIQUE'])
            turpe_tariffs['CG_BT_INF36']['UNIQUE'] = st.number_input("CG BT <= 36kVA - UNIQUE", value=turpe_tariffs['CG_BT_INF36']['UNIQUE'])

        with st.expander("Composantes de Comptage (€/an)"):
            turpe_tariffs['CC_HTA'] = st.number_input("CC HTA", value=turpe_tariffs['CC_HTA'])
            turpe_tariffs['CC_BT_SUP36'] = st.number_input("CC BT > 36kVA", value=turpe_tariffs['CC_BT_SUP36'])
            turpe_tariffs['CC_BT_INF36'] = st.number_input("CC BT <= 36kVA", value=turpe_tariffs['CC_BT_INF36'])

        with st.expander("Tarifs Puissance HTA - CU (€/kW/an)"):
            for period, value in turpe_tariffs['HTA_PUISSANCE']['CU_FIXE'].items():
                turpe_tariffs['HTA_PUISSANCE']['CU_FIXE'][period] = st.number_input(f"HTA CU - {period}", value=value, key=f"turpe_hta_cu_p_{period}")

        with st.expander("Tarifs Énergie HTA - CU (c€/kWh)"):
            for period, value in turpe_tariffs['HTA_ENERGIE']['CU_FIXE'].items():
                turpe_tariffs['HTA_ENERGIE']['CU_FIXE'][period] = st.number_input(f"HTA CU Énergie - {period}", value=value, key=f"turpe_hta_cu_e_{period}")

        with st.expander("Tarifs Puissance HTA - LU (€/kW/an)"):
            for period, value in turpe_tariffs['HTA_PUISSANCE']['LU_FIXE'].items():
                turpe_tariffs['HTA_PUISSANCE']['LU_FIXE'][period] = st.number_input(f"HTA LU - {period}", value=value, key=f"turpe_hta_lu_p_{period}")

        with st.expander("Tarifs Énergie HTA - LU (c€/kWh)"):
            for period, value in turpe_tariffs['HTA_ENERGIE']['LU_FIXE'].items():
                turpe_tariffs['HTA_ENERGIE']['LU_FIXE'][period] = st.number_input(f"HTA LU Énergie - {period}", value=value, key=f"turpe_hta_lu_e_{period}")

        with st.expander("Tarifs Puissance BT > 36kVA - CU (€/kW/an)"):
            for period, value in turpe_tariffs['BT_SUP36_PUISSANCE']['CU'].items():
                turpe_tariffs['BT_SUP36_PUISSANCE']['CU'][period] = st.number_input(f"BT>36 CU - {period}", value=value, key=f"turpe_bt36_cu_p_{period}")

        with st.expander("Tarifs Énergie BT > 36kVA - CU (c€/kWh)"):
            for period, value in turpe_tariffs['BT_SUP36_ENERGIE']['CU'].items():
                turpe_tariffs['BT_SUP36_ENERGIE']['CU'][period] = st.number_input(f"BT>36 CU Énergie - {period}", value=value, key=f"turpe_bt36_cu_e_{period}")

        with st.expander("Tarifs Puissance BT > 36kVA - LU (€/kW/an)"):
            for period, value in turpe_tariffs['BT_SUP36_PUISSANCE']['LU'].items():
                turpe_tariffs['BT_SUP36_PUISSANCE']['LU'][period] = st.number_input(f"BT>36 LU - {period}", value=value, key=f"turpe_bt36_lu_p_{period}")

        with st.expander("Tarifs Énergie BT > 36kVA - LU (c€/kWh)"):
            for period, value in turpe_tariffs['BT_SUP36_ENERGIE']['LU'].items():
                turpe_tariffs['BT_SUP36_ENERGIE']['LU'][period] = st.number_input(f"BT>36 LU Énergie - {period}", value=value, key=f"turpe_bt36_lu_e_{period}")

        app_tariffs['TURPE'] = turpe_tariffs

        # Bouton de soumission du formulaire
        submitted = st.form_submit_button("💾 Sauvegarder l'Ensemble du Référentiel")
        if submitted:
            # st.session_state.app_tariffs est déjà mis à jour par les widgets
            try:
                with open('tariffs_config.json', 'w') as f:
                    json.dump(st.session_state.app_tariffs, f, indent=4)
                st.success("Le référentiel a été sauvegardé avec succès dans `tariffs_config.json`.")
                st.rerun() # Pour s'assurer que l'UI reflète les nouvelles valeurs par défaut
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde du référentiel : {e}")

def quotation_parameters_tab():
    """Onglet des paramètres de cotation et client - VERSION OPTIMISÉE"""
    st.header("🏢 Paramètres de la Cotation")

    # --- Section 1: Informations Client ---
    st.subheader("👤 Informations Client")

    client_col1, client_col2 = st.columns(2)
    with client_col1:
        company_name = st.text_input(
            "Raison Sociale",
            value=st.session_state.client_info.get('company_name', ''),
            key="company_name_input"  # Clé stable
        )
        siren = st.text_input(
            "SIREN",
            value=st.session_state.client_info.get('siren', ''),
            key="siren_input"  # Clé stable
        )

    with client_col2:
        naf_codes_df = st.session_state.naf_codes
        naf_code, naf_activity, cee_eligible = None, None, None
        
        if not naf_codes_df.empty:
            # CORRECTION 1: Optimisation du selectbox NAF avec mise en cache
            # Créer les options seulement une fois par session
            if 'naf_options_cached' not in st.session_state:
                naf_options = pd.concat([
                    pd.DataFrame([{'display': "Sélectionnez un code NAF...", 'Code NAF': None}]),
                    naf_codes_df[['display', 'Code NAF']]
                ], ignore_index=True)
                st.session_state.naf_options_cached = naf_options

            naf_options = st.session_state.naf_options_cached
            
            # CORRECTION 2: Gestion stable de l'index du selectbox
            current_naf_code = st.session_state.client_info.get('naf_code')
            
            # Trouver l'index actuel de façon plus stable
            current_index = 0  # Index par défaut
            if current_naf_code:
                matching_rows = naf_codes_df[naf_codes_df['Code NAF'] == current_naf_code]
                if not matching_rows.empty:
                    current_display = matching_rows.iloc[0]['display']
                    try:
                        current_index = naf_options[naf_options['display'] == current_display].index[0]
                    except (IndexError, KeyError):
                        current_index = 0

            # CORRECTION 3: Selectbox avec clé stable et callback optimisé
            selected_display_naf = st.selectbox(
                "Code NAF",
                options=naf_options['display'],
                index=int(current_index),
                key="naf_selectbox",  # Clé stable
                help="Le code NAF principal du client. Il sera utilisé par défaut pour tous les sites."
            )

            # CORRECTION 4: Mise à jour conditionnelle pour éviter les recharges inutiles
            if selected_display_naf and selected_display_naf != "Sélectionnez un code NAF...":
                selected_naf_info = naf_codes_df[naf_codes_df['display'] == selected_display_naf].iloc[0]
                naf_code = selected_naf_info['Code NAF']
                naf_activity = selected_naf_info['Activité']
                cee_eligible = selected_naf_info['Eligible CEE']
        else:
            # Fallback si pas de codes NAF chargés
            naf_code_input = st.text_input(
                "Code NAF (fichier non chargé)", 
                value=st.session_state.client_info.get('naf_code', ''),
                key="naf_code_fallback"
            )
            naf_code = naf_code_input

    # --- Section 2: Informations de la Cotation ---
    st.subheader("📝 Informations de la Cotation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quotation_date = st.date_input(
            "Date de la cotation",
            value=st.session_state.quotation_info.get('quotation_date', datetime.now().date()),
            key="quotation_date_input"
        )
        start_date = st.date_input(
            "Date début de fourniture",
            value=st.session_state.quotation_info.get('start_date', date.today()),
            key="start_date_input"
        )
        end_date = st.date_input(
            "Date fin de fourniture",
            value=st.session_state.quotation_info.get('end_date', date.today() + timedelta(days=365)),
            key="end_date_input"
        )

    with col2:
        quotation_time = st.time_input(
            "Heure de la cotation",
            value=st.session_state.quotation_info.get('quotation_time', datetime.now().time()),
            key="quotation_time_input"
        )
        offer_types = ["Prix fixe unique", "Prix fixe par PHS"]
        offer_type = st.selectbox(
            "Type d'offre",
            options=offer_types,
            index=offer_types.index(st.session_state.quotation_info.get('offer_type', offer_types[0])),
            key="offer_type_selectbox"
        )

    # --- Section 3: Options de Cotation ---
    st.subheader("Options de Cotation")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        status = st.radio(
            "Statut de l'offre",
            options=["Indicatif", "Ferme"],
            index=["Indicatif", "Ferme"].index(st.session_state.quotation_info.get('status', 'Indicatif')),
            horizontal=True,
            key="status_radio"
        )
    with opt_col2:
        pricing_type = st.radio(
            "Type de calcul du prix",
            options=["Prix lissé", "Prix par année"],
            index=["Prix lissé", "Prix par année"].index(st.session_state.quotation_info.get('pricing_type', 'Prix lissé')),
            horizontal=True,
            key="pricing_type_radio",
            help="**Prix lissé**: un prix unique pour toute la période. **Prix par année**: un prix différent pour chaque année civile."
        )

    # --- Section 4: Validité de l'offre ferme ---
    validity_duration_hours = 0.0
    if status == "Ferme":
        st.subheader("Validité de l'Offre Ferme")
        v_col1, v_col2, v_col3 = st.columns(3)
        with v_col1:
            validity_end_date = st.date_input(
                "Date de fin de validité",
                value=st.session_state.quotation_info.get('validity_end_date', datetime.now().date()),
                key="validity_end_date_input"
            )
        with v_col2:
            validity_end_time = st.time_input(
                "Heure de fin de validité",
                value=st.session_state.quotation_info.get('validity_end_time', datetime.now().time()),
                key="validity_end_time_input"
            )

        # Calcul de la durée de validité
        quotation_dt = datetime.combine(quotation_date, quotation_time)
        validity_end_dt = datetime.combine(validity_end_date, validity_end_time)

        if quotation_dt < validity_end_dt:
            validity_duration_hours = calculate_business_hours(quotation_dt, validity_end_dt)
            with v_col3:
                st.metric("Durée de validité (heures ouvrées)", f"{validity_duration_hours:.2f}")
        else:
            st.warning("La date de fin de validité doit être postérieure à la date de cotation.")

    # --- CORRECTION 5: Bouton de Sauvegarde avec gestion optimisée ---
    if st.button("💾 Sauvegarder les Paramètres", key="save_params_button"):
        # Vérifier s'il y a réellement des changements avant de sauvegarder
        current_client_info = {
            'company_name': company_name,
            'siren': siren,
            'naf_code': naf_code,
            'naf_activity': naf_activity,
            'cee_eligible': cee_eligible
        }
        
        current_quotation_info = {
            'quotation_date': quotation_date,
            'quotation_time': quotation_time,
            'start_date': start_date,
            'end_date': end_date,
            'offer_type': offer_type,
            'pricing_type': pricing_type,
            'status': status,
        }
        
        if status == "Ferme":
            current_quotation_info.update({
                'validity_end_date': validity_end_date,
                'validity_end_time': validity_end_time,
                'validity_duration_hours': validity_duration_hours
            })

        # CORRECTION 6: Mise à jour conditionnelle pour éviter les recharges
        changes_made = False
        
        if st.session_state.client_info != current_client_info:
            st.session_state.client_info.update(current_client_info)
            changes_made = True
            
        if st.session_state.quotation_info != current_quotation_info:
            st.session_state.quotation_info.update(current_quotation_info)
            changes_made = True

        # Génération de l'ID de cotation
        if company_name and changes_made:
            duration_days = (end_date - start_date).days
            safe_company_name = re.sub(r'\W+', '', company_name.replace(' ', '_'))
            quotation_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_company_name[:10]}_{offer_type.replace(' ', '')}_{duration_days}j"
            st.session_state.quotation_info['quotation_id'] = quotation_id
            st.success(f"Paramètres sauvegardés avec succès! ID de cotation : {quotation_id}")
        elif not company_name:
            st.warning("Veuillez renseigner une raison sociale pour générer un ID de cotation.")
        elif not changes_made:
            st.info("Aucune modification détectée.")

    # --- Affichage des paramètres actuels ---
    st.subheader("Paramètres Actuels")

    if st.session_state.client_info.get('company_name'):
        st.write("Client:")
        client_df = pd.DataFrame([st.session_state.client_info])
        st.dataframe(client_df, use_container_width=True)

        st.write("Cotation:")
        quotation_df_display = st.session_state.quotation_info.copy()
        if 'quotation_time' in quotation_df_display and hasattr(quotation_df_display['quotation_time'], 'strftime'):
            quotation_df_display['quotation_time'] = quotation_df_display['quotation_time'].strftime('%H:%M:%S')

        quotation_df = pd.DataFrame([quotation_df_display])
        st.dataframe(quotation_df, use_container_width=True)

def sites_data_tab():
    """Onglet de gestion des données sites avec affichage des caractéristiques TURPE."""
    st.header("📊 Données des Sites")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Fichier Périmètre")
        st.info("Format accepté : fichier C68 d'ENEDIS")
        
        uploaded_perimeter = st.file_uploader(
            "Chargez le fichier CSV du périmètre ENEDIS",
            type=['csv'],
            help="Le système détecte automatiquement les colonnes standards ENEDIS"
        )
        
        if uploaded_perimeter:
            df = load_enedis_perimeter(uploaded_perimeter)
            if df is not None:
                st.session_state.perimeter_df = df

        if st.session_state.perimeter_df is not None:
            df_perimeter = st.session_state.perimeter_df
            st.subheader("Caractéristiques des Sites (TURPE)")
            turpe_calc = TURPECalculator(st.session_state.app_tariffs['TURPE'])

            for _, site_row in df_perimeter.iterrows():
                with st.expander(f"Site PRM : {site_row['prm']} ({site_row.get('company_name', 'N/A')})"):
                    site_powers = turpe_calc.get_current_powers(site_row)
                    site_time_ranges = parse_time_ranges(site_row)

                    # Formatage des puissances pour un affichage propre
                    powers_str_parts = []
                    for period, power in site_powers.items():
                        powers_str_parts.append(f"{period}: {power} kW")
                    powers_str = " | ".join(powers_str_parts) if powers_str_parts else "Non spécifiées"

                    # Formatage des plages horaires
                    peak_str = "; ".join([f"{s}h-{e}h" for s, e in site_time_ranges.get('P', [])]) or "Aucune"
                    offpeak_str = "; ".join([f"{s}h-{e}h" for s, e in site_time_ranges.get('HC', [])]) or "Aucune"

                    details_data = {
                        "Caractéristique": [
                            "Segment TURPE", "Domaine de Tension", "Formule Tarifaire",
                            "Puissances Souscrites", "Plages Heures Pleines", "Plages Heures Creuses"
                        ],
                        "Valeur": [
                            site_row.get('segment', 'N/A'),
                            site_row.get('tension_level', 'N/A'),
                            site_row.get('Formule tarifaire acheminement', 'N/A'),
                            powers_str,
                            peak_str,
                            offpeak_str
                        ]
                    }
                    details_df = pd.DataFrame(details_data)
                    st.dataframe(details_df, hide_index=True, use_container_width=True)

            # Carte des sites
            if 'latitude' in df_perimeter.columns and 'longitude' in df_perimeter.columns:
                df_map = df_perimeter.copy()
                df_map['lat'] = pd.to_numeric(df_map['latitude'], errors='coerce')
                df_map['lon'] = pd.to_numeric(df_map['longitude'], errors='coerce')
                df_map = df_map.dropna(subset=['lat', 'lon'])
                
                if not df_map.empty:
                    st.subheader("Localisation des Sites")
                    st.map(df_map[['lat', 'lon']])
    
    with col2:
        st.subheader("2. Fichiers de Consommation")
        st.info("Format accepté : fichier R63 d'ENEDIS")
        
        if st.session_state.perimeter_df is not None:
            available_prms = st.session_state.perimeter_df['prm'].unique()
            
            # Option de chargement global ou par PRM
            upload_mode = st.radio(
                "Mode de chargement",
                ["Un fichier par PRM", "Fichier unique multi-PRM"]
            )
            
            if upload_mode == "Fichier unique multi-PRM":
                uploaded_consumption_all = st.file_uploader(
                    "Fichier de consommation global",
                    type=['csv'],
                    help="Fichier contenant tous les PRMs avec colonnes : PRM, datetime, consommation"
                )
                
                if uploaded_consumption_all:
                    df_consumption = load_enedis_consumption(uploaded_consumption_all)
                    if df_consumption is not None:
                        # Séparation par PRM
                        for prm in df_consumption['prm'].unique():
                            prm_data = df_consumption[df_consumption['prm'] == prm].copy()
                            st.session_state.consumption_data[str(prm)] = prm_data
                        
                        st.success(f"Consommations chargées pour {len(df_consumption['prm'].unique())} PRMs")
                        
                        # Affichage d'un échantillon par PRM
                        for prm in df_consumption['prm'].unique():
                            prm_data = df_consumption[df_consumption['prm'] == prm]
                            
                            with st.expander(f"PRM {prm} - {len(prm_data)} points"):
                                col_prm1, col_prm2 = st.columns(2)
                                
                                with col_prm1:
                                    st.metric("Points de données", len(prm_data))
                                    st.metric("Période début", prm_data['datetime'].min().strftime('%Y-%m-%d'))
                                    st.metric("Période fin", prm_data['datetime'].max().strftime('%Y-%m-%d'))
                                
                                with col_prm2:
                                    st.metric("Puissance moyenne", f"{prm_data['power_w'].mean():.2f} W")
                                    st.metric("Puissance max", f"{prm_data['power_w'].max():.2f} W")
                                
                                # Graphique rapide
                                if len(prm_data) > 0:
                                    last_date = prm_data['datetime'].max()
                                    start_date_filter = last_date - pd.DateOffset(months=1)
                                    monthly_data = prm_data[prm_data['datetime'] >= start_date_filter]

                                    fig = px.line(
                                        monthly_data,
                                        x='datetime',
                                        y='power_w',
                                        title=f"Aperçu Puissance {prm} (dernier mois)"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
            
            else:  # Un fichier par PRM
                for prm in available_prms[:5]:  # Limiter l'affichage pour éviter la surcharge
                    with st.expander(f"PRM: {prm}"):
                        uploaded_consumption = st.file_uploader(
                            f"Consommation pour {prm}",
                            type=['csv'],
                            key=f"consumption_{prm}",
                            help="Format ENEDIS : horodate, valeur, etc."
                        )
                        
                        if uploaded_consumption:
                            df_consumption = load_enedis_consumption(uploaded_consumption)
                            if df_consumption is not None:
                                st.session_state.consumption_data[prm] = df_consumption
                                
                                # Aperçu graphique
                                last_date = df_consumption['datetime'].max()
                                start_date_filter = last_date - pd.DateOffset(months=1)
                                monthly_data = df_consumption[df_consumption['datetime'] >= start_date_filter]

                                fig = px.line(
                                    monthly_data,
                                    x='datetime',
                                    y='power_w',
                                    title=f"Aperçu puissance {prm} (dernier mois)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Veuillez d'abord charger le fichier périmètre")
    
    # Validation de cohérence des données
    if st.session_state.perimeter_df is not None and st.session_state.consumption_data:
        st.subheader("Validation de Cohérence des Données")
        validation_results = validate_data_consistency(
            st.session_state.perimeter_df, 
            st.session_state.consumption_data
        )
        
        # Affichage des statistiques
        if validation_results['stats']:
            col_val1, col_val2, col_val3, col_val4 = st.columns(4)
            
            with col_val1:
                st.metric("Sites Périmètre", validation_results['stats']['perimeter_sites'])
            with col_val2:
                st.metric("Sites avec Consommation", validation_results['stats']['consumption_sites'])
            with col_val3:
                st.metric("Sites Communs", validation_results['stats']['common_sites'])
            with col_val4:
                st.metric("Taux de Couverture", f"{validation_results['stats']['common_sites']/validation_results['stats']['perimeter_sites']*100:.1f}%")
        
        # Affichage des warnings
        for warning in validation_results['warnings']:
            st.warning(warning)
        
        # Affichage des erreurs
        for error in validation_results['errors']:
            st.error(error)


def site_details_tab():
    """Onglet pour la gestion des détails par site."""
    st.header("✍️ Détail par Site")

    if st.session_state.get('perimeter_df') is None or st.session_state.perimeter_df.empty:
        st.warning("Veuillez d'abord charger un fichier de périmètre dans l'onglet 'Données Sites'.")
        return

    st.info("Cette section vous permet de surcharger les paramètres globaux pour des sites spécifiques. Les valeurs modifiées ici prendront le pas sur les paramètres de la cotation.")

    # Préparation des données pour l'éditeur
    sites_df = st.session_state.perimeter_df[['prm']].copy()
    sites_df['company_name'] = st.session_state.client_info.get('company_name', 'Non renseigné')


    # Récupérer les valeurs globales
    global_start_date = st.session_state.quotation_info.get('start_date', date.today())
    global_end_date = st.session_state.quotation_info.get('end_date', date.today() + timedelta(days=365))
    global_naf_code = st.session_state.client_info.get('naf_code', '')

    # Appliquer les surcharges existantes ou les valeurs globales
    details_list = []
    for _, row in sites_df.iterrows():
        prm = row['prm']
        site_specific_details = st.session_state.site_details.get(str(prm), {})
        details_list.append({
            'start_date': site_specific_details.get('start_date', global_start_date),
            'end_date': site_specific_details.get('end_date', global_end_date),
            'naf_code': site_specific_details.get('naf_code', global_naf_code)
        })

    details_df = pd.DataFrame(details_list)
    editor_df = pd.concat([sites_df.reset_index(drop=True), details_df.reset_index(drop=True)], axis=1)

    st.subheader("Configuration par site")

    # S'assurer que les codes NAF sont chargés
    if 'naf_codes' not in st.session_state or st.session_state.naf_codes.empty:
        st.error("La liste des codes NAF n'a pas pu être chargée. Impossible de configurer par site.")
        return

    valid_naf_codes = st.session_state.naf_codes['Code NAF'].tolist()

    # Utiliser st.data_editor pour permettre la modification
    edited_df = st.data_editor(
        editor_df,
        column_config={
            "prm": st.column_config.TextColumn("PRM", disabled=True, help="Point de Référence Mesure (non modifiable)"),
            "company_name": st.column_config.TextColumn("Raison Sociale", disabled=True, help="Nom de l'entreprise (non modifiable)"),
            "start_date": st.column_config.DateColumn("Date de début", format="DD/MM/YYYY", required=True),
            "end_date": st.column_config.DateColumn("Date de fin", format="DD/MM/YYYY", required=True),
            "naf_code": st.column_config.SelectboxColumn(
                "Code NAF",
                options=valid_naf_codes,
                required=True,
                help="Code NAF spécifique pour ce site"
            )
        },
        hide_index=True,
        use_container_width=True,
        key="site_details_editor"
    )

    if st.button("💾 Sauvegarder les modifications"):
        new_site_details = {}
        for _, row in edited_df.iterrows():
            prm = str(row['prm'])

            # Vérifier si les dates ou le code NAF sont différents des valeurs globales
            is_different = (
                pd.to_datetime(row['start_date']).date() != global_start_date or
                pd.to_datetime(row['end_date']).date() != global_end_date or
                row['naf_code'] != global_naf_code
            )

            if is_different:
                new_site_details[prm] = {
                    'start_date': pd.to_datetime(row['start_date']).date(),
                    'end_date': pd.to_datetime(row['end_date']).date(),
                    'naf_code': row['naf_code']
                }

        # Comparer avec l'ancien état pour voir s'il y a eu un changement
        if new_site_details != st.session_state.get('site_details', {}):
            st.session_state.site_details = new_site_details
            st.success(f"Modifications sauvegardées ! {len(new_site_details)} site(s) ont maintenant des paramètres spécifiques.")
            st.rerun()
        else:
            st.info("Aucune nouvelle modification détectée.")

    # Afficher les surcharges actuelles pour vérification
    if st.session_state.site_details:
        st.subheader("Surcharges Actives")

        # Convertir les objets date en string pour l'affichage JSON
        display_details = {}
        for prm, details in st.session_state.site_details.items():
            display_details[prm] = {
                'start_date': details['start_date'].strftime('%Y-%m-%d'),
                'end_date': details['end_date'].strftime('%Y-%m-%d'),
                'naf_code': details['naf_code']
            }

        st.json(display_details, expanded=False)


def price_curve_tab():
    """Onglet de la courbe de prix - VERSION SIMPLIFIÉE (import CSV uniquement)"""
    st.header("💰 Courbe de Prix")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration")
        
        time_step = st.selectbox(
            "Pas de temps",
            options=["Horaire", "15 minutes"],
            index=0
        )
        
        # Import CSV uniquement
        uploaded_price_file = st.file_uploader(
            "Fichier courbe de prix",
            type=['csv'],
            help="Format attendu : datetime, prix_euro_mwh"
        )
        
        if uploaded_price_file:
            # Essayer différentes configurations
            configs = [
                {'sep': ',', 'encoding': 'utf-8'},
                {'sep': ';', 'encoding': 'utf-8'},
                {'sep': ',', 'encoding': 'latin-1'},
                {'sep': ';', 'encoding': 'latin-1'},
                {'sep': None, 'encoding': 'utf-8'},  # Auto-détection
            ]
            
            df_price = None
            for config in configs:
                try:
                    df_price = pd.read_csv(uploaded_price_file, **{k: v for k, v in config.items() if v is not None})
                    # Vérifier que nous avons bien 2 colonnes
                    if df_price.shape[1] >= 2:
                        break
                except:
                    continue
            
            if df_price is None or df_price.shape[1] < 2:
                st.error("Impossible de lire le fichier CSV. Vérifiez le format et l'encodage.")
            else:
                try:
                    # Standardisation des colonnes
                    df_price.columns = df_price.columns.str.lower().str.strip()
                    
                    # Mapping des colonnes de prix
                    price_mapping = {
                        'prix': 'prix_euro_mwh',
                        'price': 'prix_euro_mwh',
                        'valeur': 'prix_euro_mwh',
                        'value': 'prix_euro_mwh',
                        'horodate': 'datetime',
                        'date_heure': 'datetime',
                        'timestamp': 'datetime'
                    }
                    
                    for old_name, new_name in price_mapping.items():
                        if old_name in df_price.columns:
                            df_price = df_price.rename(columns={old_name: new_name})
                    
                    df_price['datetime'] = pd.to_datetime(df_price['datetime'])
                    df_price['prix_euro_mwh'] = pd.to_numeric(df_price['prix_euro_mwh'], errors='coerce')
                    df_price = df_price.dropna()
                    
                    st.session_state.price_curve_df = df_price
                    st.success(f"Courbe de prix chargée: {len(df_price)} points")
                except Exception as e:
                    st.error(f"Erreur lors du traitement: {e}")
    
    with col2:
        st.subheader("Visualisation")
        
        if st.session_state.price_curve_df is not None:
            df_price = st.session_state.price_curve_df
            
            # Graphique principal
            fig = px.line(
                df_price,
                x='datetime',
                y='prix_euro_mwh',
                title="Courbe de Prix de l'Électricité"
            )
            fig.update_layout(
                xaxis_title="Date/Heure",
                yaxis_title="Prix (€/MWh)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.metric("Prix Moyen", f"{df_price['prix_euro_mwh'].mean():.2f} €/MWh")
            with col_stat2:
                st.metric("Prix Min", f"{df_price['prix_euro_mwh'].min():.2f} €/MWh")
            with col_stat3:
                st.metric("Prix Max", f"{df_price['prix_euro_mwh'].max():.2f} €/MWh")
            
            # Aperçu des données
            st.dataframe(df_price.head(24), use_container_width=True)
        else:
            st.info("Aucune courbe de prix chargée")

def risk_premiums_tab():
    """Onglet de configuration des paramètres tarifaires pour la cotation en cours."""
    st.header("⚠️ Paramètres Tarifaires de la Cotation")
    st.info("Cette section permet de surcharger les valeurs par défaut du Référentiel pour cette cotation spécifique.")

    # Récupérer les valeurs par défaut du référentiel
    defaults = st.session_state.app_tariffs

    # --- Section 1: Marge Commerciale ---
    st.subheader("Marge Commerciale")
    default_marge = defaults.get('Marge', {}).get('marge_commerciale', 5.0)
    marge_commerciale = st.number_input(
        "Marge commerciale (€/MWh)",
        value=float(st.session_state.tariff_params.get('marge_commerciale', default_marge)),
        min_value=0.0, step=0.1, format="%.2f",
        key="marge_commerciale_input",
        help=f"Valeur par défaut du référentiel : {default_marge:.2f} €/MWh"
    )
    st.session_state.tariff_params['marge_commerciale'] = marge_commerciale
    st.markdown("---")

    # --- Section 2: CEE, Capacité et GO (avec surcharges annuelles) ---
    st.subheader("CEE, Capacité et GO")
    default_couts = defaults.get('Autres_Couts', {})

    start_date = st.session_state.quotation_info.get('start_date', date.today())
    end_date = st.session_state.quotation_info.get('end_date', date.today() + timedelta(days=365))

    if start_date > end_date:
        st.error("La date de début de fourniture ne peut pas être après la date de fin.")
        return

    years = list(range(start_date.year, end_date.year + 1))

    with st.expander("Coûts CEE par année (€/MWh)", expanded=True):
        default_cee = default_couts.get('CEE', 12.0)
        st.caption(f"Valeur par défaut du référentiel : {default_cee:.2f} €/MWh")
        cee_costs = st.session_state.tariff_params.get('cee_costs_yearly', {})
        new_cee_costs = {}
        cols = st.columns(len(years) if years else 1)
        for i, year in enumerate(years):
            with cols[i]:
                new_cee_costs[str(year)] = st.number_input(
                    f"CEE {year}",
                    value=float(cee_costs.get(str(year), default_cee)),
                    min_value=0.0, step=0.1, format="%.2f", key=f"cee_{year}"
                )
        st.session_state.tariff_params['cee_costs_yearly'] = new_cee_costs

    with st.expander("Coûts de Capacité par année (€/MWh)", expanded=True):
        default_capa = default_couts.get('Garantie_Capacite', 3.0)
        st.caption(f"Valeur par défaut du référentiel : {default_capa:.2f} €/MWh")
        capacity_costs = st.session_state.tariff_params.get('capacity_costs_yearly', {})
        new_capacity_costs = {}
        cols_cap = st.columns(len(years) if years else 1)
        for i, year in enumerate(years):
            with cols_cap[i]:
                new_capacity_costs[str(year)] = st.number_input(
                    f"Capacité {year}",
                    value=float(capacity_costs.get(str(year), default_capa)),
                    min_value=0.0, step=0.1, format="%.2f", key=f"capacity_{year}"
                )
        st.session_state.tariff_params['capacity_costs_yearly'] = new_capacity_costs

    st.markdown("---")

    # --- Section 3: Option Electricité Verte ---
    st.subheader("Option Electricité Verte")
    green_option_activated = st.checkbox(
        "Activer l'option électricité verte",
        value=st.session_state.tariff_params.get('green_option_activated', False),
        key='green_option_toggle'
    )
    st.session_state.tariff_params['green_option_activated'] = green_option_activated

    if green_option_activated:
        with st.expander("Coûts des Garanties d'Origine par année (€/MWh)", expanded=True):
            default_go = default_couts.get('Garanties_Origine', 0.8)
            st.caption(f"Valeur par défaut du référentiel : {default_go:.2f} €/MWh")
            go_costs = st.session_state.tariff_params.get('go_costs_yearly', {})
            new_go_costs = {}
            cols_go = st.columns(len(years) if years else 1)
            for i, year in enumerate(years):
                with cols_go[i]:
                    new_go_costs[str(year)] = st.number_input(
                        f"GO {year}",
                        value=float(go_costs.get(str(year), default_go)),
                        min_value=0.0, step=0.1, format="%.2f", key=f"go_{year}"
                    )
            st.session_state.tariff_params['go_costs_yearly'] = new_go_costs

    st.markdown("---")

    # --- Section 4: Configuration des Primes de Risque pour la Cotation ---
    st.subheader("Primes de Risque de la Cotation")
    st.info("Les valeurs initiales sont chargées depuis le Référentiel. Vous pouvez les surcharger ici pour cette cotation spécifique.")

    # S'assurer que la structure de configuration existe
    if 'risk_premiums_config' not in st.session_state:
        st.session_state.risk_premiums_config = {
            'default_premiums': defaults.get('Primes_Risques', {}).copy()
        }
    
    # Utiliser une copie pour éviter de modifier directement le dictionnaire de la session pendant le rendu
    quote_premiums = st.session_state.risk_premiums_config['default_premiums']

    tab1, tab2, tab3 = st.tabs(["Primes de la Cotation", "Import Spécifique par Site", "Calcul Automatique"])

    with tab1:
        default_premiums_ref = defaults.get('Primes_Risques', {})
        pr_col1, pr_col2 = st.columns(2)

        # Fonction pour générer les inputs et éviter la répétition
        def premium_input(name, key):
            default_val = default_premiums_ref.get(name, 0.0)
            return st.number_input(
                f"Prime: {name}",
                value=float(quote_premiums.get(name, default_val)),
                min_value=0.0, step=0.1, format="%.2f", key=key,
                help=f"Défaut du référentiel : {default_val:.2f}"
            )

        with pr_col1:
            quote_premiums["Délai de validité"] = premium_input("Délai de validité", "quote_p_validite")
            quote_premiums["Coût d'équilibrage"] = premium_input("Coût d'équilibrage", "quote_p_equilibrage")
            quote_premiums["Coût du profil"] = premium_input("Coût du profil", "quote_p_profil")
            quote_premiums["Risque volume"] = premium_input("Risque volume", "quote_p_volume")
        with pr_col2:
            quote_premiums["Fluctuation du parc"] = premium_input("Fluctuation du parc", "quote_p_fluc")
            quote_premiums["Recalage Prix Live"] = premium_input("Recalage Prix Live", "quote_p_recalage")
            quote_premiums["Délai de Paiement"] = premium_input("Délai de Paiement", "quote_p_paiement")
            quote_premiums["Risque Crédit"] = premium_input("Risque Crédit", "quote_p_credit")

        st.session_state.risk_premiums_config['default_premiums'] = quote_premiums

    with tab2:
        st.subheader("Surcharge des Primes par Site (Optionnel)")
        
        # NOUVEAU : Info-bulle détaillée avec format attendu
        help_text = """
        Le fichier CSV doit contenir les colonnes suivantes :
        
        **Colonnes obligatoires :**
        - `site_id` ou `prm` : Identifiant du site (PRM)
        
        **Colonnes optionnelles (primes en €/MWh) :**
        - `Délai de validité`
        - `Coût d'équilibrage`
        - `Coût du profil`
        - `Risque volume`
        - `Fluctuation du parc`
        - `Recalage Prix Live`
        - `Délai de Paiement`
        - `Risque Crédit`
        
        **Format :** CSV avec séparateur `;` ou `,`
        
        Les primes non renseignées pour un site utiliseront les valeurs par défaut configurées dans l'onglet "Primes de la Cotation".
        """
        
        uploaded_premiums_file = st.file_uploader(
            "Importer un CSV de primes par site",
            type=['csv'],
            help="Format : PRM (texte), puis colonnes optionnelles pour chaque prime (nombres)",
            key="quote_premiums_uploader"
        )
        
        if uploaded_premiums_file:
            try:
                # Lecture avec gestion des séparateurs
                try:
                    premiums_df = pd.read_csv(uploaded_premiums_file, sep=';', encoding='utf-8')
                except:
                    uploaded_premiums_file.seek(0)
                    try:
                        premiums_df = pd.read_csv(uploaded_premiums_file, sep=',', encoding='utf-8')
                    except:
                        uploaded_premiums_file.seek(0)
                        premiums_df = pd.read_csv(uploaded_premiums_file, sep=';', encoding='latin-1')
                
                # CORRECTION 1: Nettoyage des noms de colonnes
                premiums_df.columns = premiums_df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
                
                # st.write("DEBUG - Colonnes détectées après nettoyage:")
                # st.write(list(premiums_df.columns))
                
                # CORRECTION 2: Mapping et conversion des types
                column_mapping = {
                    'site_id': 'PRM',
                    'prm': 'PRM', 
                    'identifiant': 'PRM'
                }
                
                for old_name, new_name in column_mapping.items():
                    if old_name in premiums_df.columns:
                        premiums_df = premiums_df.rename(columns={old_name: new_name})
                        break
                
                # CORRECTION 3: Conversion forcée de PRM en string
                if 'PRM' in premiums_df.columns:
                    # Conversion en string avec gestion des NaN
                    premiums_df['PRM'] = premiums_df['PRM'].astype(str).str.strip()
                    # Supprimer les lignes avec PRM vide ou 'nan'
                    premiums_df = premiums_df[~premiums_df['PRM'].isin(['', 'nan', 'None'])]
                    
                    # st.write("DEBUG - Types après conversion:")
                    # st.write(premiums_df.dtypes)
                    
                    # st.write("DEBUG - Échantillon des PRMs convertis:")
                    # st.write(premiums_df['PRM'].head().tolist())
                    
                    # CORRECTION 4: Validation des colonnes de primes
                    expected_premium_columns = [
                        'Délai de validité', 'Coût d\'équilibrage', 'Coût du profil',
                        'Risque volume', 'Fluctuation du parc', 'Recalage Prix Live',
                        'Délai de Paiement', 'Risque Crédit'
                    ]
                    
                    found_premiums = []
                    for col in expected_premium_columns:
                        if col in premiums_df.columns:
                            found_premiums.append(col)
                    
                    # CORRECTION 5: Conversion des colonnes numériques
                    for col in found_premiums:
                        premiums_df[col] = pd.to_numeric(premiums_df[col], errors='coerce')
                    
                    if found_premiums:
                        st.session_state.premiums_df = premiums_df
                        st.success(f"Fichier de primes spécifiques chargé pour {len(premiums_df)} sites.")
                        st.info(f"Primes trouvées : {', '.join(found_premiums)}")
                        
                        # Test de correspondance avec le périmètre
                        if st.session_state.get('perimeter_df') is not None:
                            perimeter_prms = set(st.session_state.perimeter_df['prm'].astype(str).str.strip())
                            csv_prms = set(premiums_df['PRM'].str.strip())
                            
                            matches = perimeter_prms & csv_prms
                            st.success(f"Correspondances trouvées : {len(matches)} sites")
                            
                            if len(matches) == 0:
                                st.warning("ATTENTION: Aucune correspondance entre les PRMs du CSV et du périmètre!")
                                st.write("Premiers PRMs du périmètre:", list(perimeter_prms)[:3])
                                st.write("Premiers PRMs du CSV:", list(csv_prms)[:3])
                        
                        # Aperçu final
                        display_columns = ['PRM'] + found_premiums
                        st.dataframe(premiums_df[display_columns].head(), use_container_width=True)
                    else:
                        st.warning("Aucune colonne de prime reconnue dans le fichier.")
                        st.info(f"Colonnes attendues : {', '.join(expected_premium_columns)}")
                else:
                    st.error("Erreur : Aucune colonne d'identifiant trouvée (PRM, site_id, etc.)")
                        
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier de primes : {e}")
                st.exception(e)

    with tab3:
        st.subheader("Calcul Automatique de la Prime 'Délai de Validité'")
        calculator = RiskPremiumCalculator()

        col_calc1, col_calc2 = st.columns(2)
        with col_calc1:
            validity_duration = st.number_input(
                "Durée de validité (heures)",
                value=st.session_state.quotation_info.get('validity_duration_hours', 0.0),
                min_value=0.0, step=0.1, format="%.2f",
                help="Durée calculée si l'offre est 'Ferme', sinon modifiable manuellement."
            )
            market_price = st.number_input(
                "Prix de marché de référence (€/MWh)",
                value=60.0, min_value=0.0, step=1.0,
                help="Prix spot ou forward de référence du marché électrique"
            )
        with col_calc2:
            volatility_pct = st.number_input(
                "Volatilité annuelle (%)", value=25.0, min_value=0.0, step=1.0,
                help="Volatilité historique annualisée des prix (généralement 20-50%)"
            )
            risk_coverage = st.number_input(
                "Niveau de risque à couvrir (%)", value=95.0, min_value=50.0, max_value=99.9, step=0.1,
                help="Niveau de confiance VaR (95% = couverture de 95% des cas défavorables)"
            )

        if st.button("Calculer et Appliquer la Prime de Validité", type="primary"):
            with st.spinner("Calcul en cours..."):
                calc_result = calculator.calculate_validity_period_premium(
                    validity_hours=validity_duration,
                    market_price=market_price,
                    volatility_pct=volatility_pct,
                    risk_coverage_pct=risk_coverage,
                    price_curve=st.session_state.get('price_curve_df')
                )
                calculated_premium = calc_result['premium_eur_mwh']
                # Mettre à jour la prime dans la configuration de la cotation
                st.session_state.risk_premiums_config['default_premiums']["Délai de validité"] = calculated_premium
                st.success(f"Prime calculée ({calculated_premium:.3f} €/MWh) et appliquée. Voir l'onglet 'Primes de la Cotation'.")
                st.rerun()

    # Affichage des primes actuellement configurées pour la cotation
    st.subheader("Configuration Actuelle des Primes pour la Cotation")
    current_premiums = st.session_state.risk_premiums_config['default_premiums']
    breakdown_df = RiskPremiumCalculator().get_premium_breakdown_df(current_premiums)
    st.dataframe(breakdown_df.style.format({'Valeur (€/MWh)': '{:.3f}'}), use_container_width=True)
    fig_breakdown = px.pie(
        breakdown_df[breakdown_df['Type de Prime'] != 'TOTAL PRIMES'],
        values='Valeur (€/MWh)', names='Type de Prime',
        title="Répartition des Primes de Risque pour la Cotation"
    )
    st.plotly_chart(fig_breakdown, use_container_width=True)

def forecast_tab():
    """Onglet de prévision de consommation avec option multi-sites."""
    st.header("🔮 Prévision de Consommation")
    
    if not st.session_state.consumption_data:
        st.warning("Veuillez d'abord charger les données de consommation")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        available_sites = list(st.session_state.consumption_data.keys())
        
        # Paramètres du modèle
        st.subheader("Paramètres XGBoost")
        time_step_forecast = st.selectbox(
            "Granularité de la prévision",
            options=["Horaire", "15 minutes"],
            index=0
        )
        test_size = st.slider("Taille de l'échantillon test (%)", 10, 40, 20) / 100
        n_estimators = st.number_input("Nombre d'estimateurs", 50, 500, 100)
        
        st.subheader("Actions")

        # Sélection du site pour l'analyse et l'affichage
        selected_site_for_display = st.selectbox(
            "Site à afficher dans les résultats",
            available_sites,
            help="Sélectionnez un site pour visualiser ses résultats détaillés ci-contre."
        )

        if st.button("🚀 Lancer Prévision (tous les sites)"):
            total_sites = len(available_sites)
            if total_sites > 0:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, site_id in enumerate(available_sites):
                    status_text.text(f"Prévision en cours pour le site {i+1}/{total_sites} : {site_id}")
                    # On passe les mêmes paramètres à chaque site pour la cohérence
                    forecast_consumption(site_id, test_size, n_estimators, time_step_forecast)
                    progress_bar.progress((i + 1) / total_sites)

                status_text.success(f"Prévisions terminées pour les {total_sites} sites !")
                st.balloons()
            else:
                st.warning("Aucun site avec des données de consommation n'a été chargé.")

    with col2:
        st.subheader("Résultats")
        
        # Affichage des résultats pour le site sélectionné
        if selected_site_for_display and selected_site_for_display in st.session_state.forecast_results:
            results = st.session_state.forecast_results[selected_site_for_display]
            
            st.info(f"Affichage des résultats pour le site : **{selected_site_for_display}**")

            # Métriques du modèle
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("R² Score", f"{results['r2_score']:.3f}")
            with col_met2:
                st.metric("MAE", f"{results['mae']:.2f} W")
            
            # Graphique de prévision
            if 'forecast_df' in results:
                fig = px.line(
                    results['forecast_df'],
                    x='datetime',
                    y='predicted_power_w',
                    title=f"Prévision de Puissance - {selected_site_for_display}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Importance des features
            if 'feature_importance' in results:
                st.subheader("Importance des Variables")
                importance_df = results['feature_importance']
                fig_imp = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Importance des Features"
                )
                st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("Aucune prévision n'a encore été calculée ou aucun site n'est sélectionné pour l'affichage.")

def forecast_consumption(selected_site, test_size, n_estimators, time_step_forecast):
    """Lance la prévision de consommation pour un site avec validation robuste des données"""
    
    with st.spinner(f"Préparation des données pour {selected_site} (granularité: {time_step_forecast})..."):
        
        # Mapping de la granularité
        freq_map = {"Horaire": "H", "15 minutes": "15T"}
        minutes_map = {"Horaire": 60, "15 minutes": 15}
        freq = freq_map[time_step_forecast]
        time_step_minutes = minutes_map[time_step_forecast]

        # Récupération des données de consommation
        df_power = st.session_state.consumption_data[selected_site].copy()
        
        # Vérification initiale des données
        if df_power.empty:
            st.error(f"[{selected_site}] Aucune donnée de consommation pour le site.")
            return

        # Contrôle de l'historique de consommation
        min_date = df_power['datetime'].min()
        max_date = df_power['datetime'].max()
        history_days = (max_date - min_date).days

        if history_days < 270:  # Moins de 9 mois
            st.warning(f"[{selected_site}] Historique de consommation insuffisant ({history_days} jours). Moins de 9 mois de données disponibles. La prévision pour ce site sera ignorée.")
            return
        elif history_days < 365:  # Entre 9 et 12 mois (non inclus)
            # NOUVEAU : Avertissement non bloquant pour 9-12 mois
            st.warning(f"[{selected_site}] Attention : Historique de consommation limité ({history_days} jours, soit {history_days/30.44:.1f} mois). La précision de la prévision pourrait être affectée par le manque de données saisonnières complètes. Recommandation : au moins 12 mois d'historique.")
        
        
        st.info(f"Données initiales : {len(df_power)} points")
        
        # CORRECTION: Rééchantillonnage avec calcul correct de l'énergie
        st.info(f"Rééchantillonnage des données en énergie ({time_step_forecast})...")
        
        # Déterminer le pas de temps original des données
        if len(df_power) > 1:
            original_step = df_power['datetime'].diff().median()
            original_minutes = original_step.total_seconds() / 60
            st.info(f"Pas de temps original détecté : {original_minutes:.0f} minutes")
        else:
            original_minutes = 5  # Défaut
        
        # Convertir la puissance en énergie pour l'intervalle original
        df_power['energy_wh'] = df_power['power_w'] * (original_minutes / 60)  # Wh
        
        # Rééchantillonnage en sommant les énergies (pas en moyennant les puissances)
        df_resampled = df_power.resample(freq, on='datetime')['energy_wh'].sum().reset_index()
        df_resampled = df_resampled.dropna(subset=['energy_wh'])
        
        # Reconvertir en puissance moyenne pour le nouvel intervalle
        df_resampled['power_w'] = df_resampled['energy_wh'] / (time_step_minutes / 60)
        
        st.info(f"Données après rééchantillonnage : {len(df_resampled)} points")

        min_points = (2 * 24 * 60) / time_step_minutes # Au moins 2 jours de données
        if len(df_resampled) < min_points:
            st.error(f"Données insuffisantes après rééchantillonnage ({len(df_resampled)} points) pour la prévision.")
            return

        # Feature engineering temporel
        df_resampled['hour'] = df_resampled['datetime'].dt.hour
        df_resampled['day_of_week'] = df_resampled['datetime'].dt.dayofweek
        df_resampled['month'] = df_resampled['datetime'].dt.month
        df_resampled['day_of_year'] = df_resampled['datetime'].dt.dayofyear

        # NOUVEAU: Features pour les jours fériés
        fr_holidays = holidays.France()
        df_resampled['is_holiday'] = df_resampled['datetime'].dt.date.isin(fr_holidays)
        df_resampled['is_day_before_holiday'] = (df_resampled['datetime'] + pd.Timedelta(days=1)).dt.date.isin(fr_holidays)
        
        # Features cycliques
        df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled['hour'] / 24)
        df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled['hour'] / 24)
        df_resampled['day_sin'] = np.sin(2 * np.pi * df_resampled['day_of_week'] / 7)
        df_resampled['day_cos'] = np.cos(2 * np.pi * df_resampled['day_of_week'] / 7)
        df_resampled['month_sin'] = np.sin(2 * np.pi * df_resampled['month'] / 12)
        df_resampled['month_cos'] = np.cos(2 * np.pi * df_resampled['month'] / 12)

        # Features de lag dynamiques
        points_in_day = 24 * 60 / time_step_minutes
        lag_24h = int(points_in_day)
        lag_7d = int(points_in_day * 7)

        if len(df_resampled) > lag_24h:
            df_resampled['power_lag_24h'] = df_resampled['power_w'].shift(lag_24h)
        else:
            df_resampled['power_lag_24h'] = df_resampled['power_w'].mean()
        
        if len(df_resampled) > lag_7d:
            df_resampled['power_lag_7d'] = df_resampled['power_w'].shift(lag_7d)
        else:
            df_resampled['power_lag_7d'] = df_resampled['power_w'].mean()
        
        # Préparation des features
        feature_cols = ['hour', 'day_of_week', 'month', 'day_of_year',
                       'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
                       'power_lag_24h', 'power_lag_7d', 'is_holiday', 'is_day_before_holiday']
        
        # Vérification de la présence de toutes les features
        missing_features = [col for col in feature_cols if col not in df_resampled.columns]
        if missing_features:
            st.error(f"Features manquantes : {missing_features}")
            return
        
        # Nettoyage des données avec validation
        initial_count = len(df_resampled)
        df_resampled = df_resampled.dropna(subset=feature_cols + ['power_w'])
        final_count = len(df_resampled)
        
        st.info(f"Données après nettoyage : {final_count} points (supprimés : {initial_count - final_count})")
        
        # Vérification finale de la taille du dataset
        if final_count < 100:
            st.error(f"Données insuffisantes après nettoyage : {final_count} points. Minimum requis : 100 points.")
            st.error("Vérifiez la qualité de vos données de consommation.")
            return
        
        # Préparation des données pour l'entraînement
        X = df_resampled[feature_cols]
        y = df_resampled['power_w']
        
        # Validation des données
        if X.isnull().sum().sum() > 0:
            st.error("Des valeurs nulles subsistent dans les features")
            st.write("Valeurs nulles par feature :")
            st.write(X.isnull().sum())
            return
        
        # Ajustement du test_size si nécessaire
        min_test_size = max(50, int(0.1 * len(X)))  # Au moins 50 points ou 10%
        max_test_points = int(test_size * len(X))
        
        if max_test_points < min_test_size:
            adjusted_test_size = min_test_size / len(X)
            st.warning(f"Test size ajusté de {test_size:.2%} à {adjusted_test_size:.2%} pour avoir suffisamment de données d'entraînement")
            test_size = adjusted_test_size
        
        # Division train/test avec stratification temporelle (pas de shuffle)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=False
            )
            
            st.info(f"Données d'entraînement : {len(X_train)} points")
            st.info(f"Données de test : {len(X_test)} points")
            
        except ValueError as e:
            st.error(f"Erreur lors de la division des données : {str(e)}")
            st.error(f"Taille du dataset : {len(X)} points")
            st.error(f"Test size demandé : {test_size:.2%}")
            return
        
        # Entraînement du modèle XGBoost
        try:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=42,
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            )
            
            model.fit(X_train, y_train)
            
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle : {str(e)}")
            return
        
        # Prédictions et métriques
        try:
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
        except Exception as e:
            st.error(f"Erreur lors des prédictions : {str(e)}")
            return
        
        # Importance des features
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Génération de la prévision sur la période de cotation
        forecast_df = None
        if st.session_state.quotation_info:
            # NOUVEAU : Récupérer les dates spécifiques au site ou les dates globales
            site_details = st.session_state.site_details.get(str(selected_site), {})
            start_date = site_details.get('start_date', st.session_state.quotation_info['start_date'])
            end_date = site_details.get('end_date', st.session_state.quotation_info['end_date'])

            st.info(f"Génération de la prévision du {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}")
            
            # CORRECTION: Création de la période de prévision jusqu'à la FIN du jour final
            try:
                # Pour inclure tout le dernier jour, on va jusqu'au lendemain minuit moins un pas
                if freq == 'H':
                    # Pour horaire : jusqu'à 23h00 du dernier jour
                    actual_end = pd.Timestamp(end_date.year, end_date.month, end_date.day, 23, 0, 0)
                else:  # freq == '15T'
                    # Pour 15 minutes : jusqu'à 23h45 du dernier jour
                    actual_end = pd.Timestamp(end_date.year, end_date.month, end_date.day, 23, 45, 0)
                
                forecast_dates = pd.date_range(start=start_date, end=actual_end, freq=freq)
                forecast_df = pd.DataFrame({'datetime': forecast_dates})
                
                st.info(f"Plage de prévision : {forecast_dates[0]} à {forecast_dates[-1]} ({len(forecast_dates)} points)")
                
                # Features temporelles pour la prévision
                forecast_df['hour'] = forecast_df['datetime'].dt.hour
                forecast_df['day_of_week'] = forecast_df['datetime'].dt.dayofweek
                forecast_df['month'] = forecast_df['datetime'].dt.month
                forecast_df['day_of_year'] = forecast_df['datetime'].dt.dayofyear
                
                # Features cycliques
                forecast_df['hour_sin'] = np.sin(2 * np.pi * forecast_df['hour'] / 24)
                forecast_df['hour_cos'] = np.cos(2 * np.pi * forecast_df['hour'] / 24)
                forecast_df['day_sin'] = np.sin(2 * np.pi * forecast_df['day_of_week'] / 7)
                forecast_df['day_cos'] = np.cos(2 * np.pi * forecast_df['day_of_week'] / 7)
                forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
                forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)

                # NOUVEAU: Features jours fériés pour la prévision
                forecast_df['is_holiday'] = forecast_df['datetime'].dt.date.isin(fr_holidays)
                forecast_df['is_day_before_holiday'] = (forecast_df['datetime'] + pd.Timedelta(days=1)).dt.date.isin(fr_holidays)
                
                # Features de lag (approximation avec moyennes historiques par heure/jour)
                historical_patterns = df_resampled.groupby(['hour', 'day_of_week'])['power_w'].mean()
                
                # Application des patterns historiques
                forecast_df['historical_pattern'] = forecast_df.apply(
                    lambda row: historical_patterns.get((row['hour'], row['day_of_week']), 
                                                       df_resampled['power_w'].mean()),
                    axis=1
                )
                
                forecast_df['power_lag_24h'] = forecast_df['historical_pattern']
                forecast_df['power_lag_7d'] = forecast_df['historical_pattern']
                
                # Prédiction
                X_forecast = forecast_df[feature_cols]
                forecast_df['predicted_power_w'] = model.predict(X_forecast)
                
                # Suppression des colonnes auxiliaires
                forecast_df = forecast_df.drop('historical_pattern', axis=1)
                
            except Exception as e:
                st.warning(f"Impossible de générer les prévisions pour la période de cotation : {str(e)}")
                forecast_df = None
        
        # Sauvegarde des résultats
        results = {
            'model': model,
            'mae': mae,
            'r2_score': r2,
            'feature_importance': importance_df,
            'data_quality': {
                'initial_points': initial_count,
                'final_points': final_count,
                'data_loss_ratio': (initial_count - final_count) / initial_count if initial_count > 0 else 0,
                'time_step_minutes': time_step_minutes,
                'granularity': time_step_forecast,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'original_step_minutes': original_minutes  # NOUVEAU: pour le debugging
            }
        }
        
        if forecast_df is not None:
            results['forecast_df'] = forecast_df

            # --- NOUVELLE FONCTIONNALITÉ: CLASSIFICATION PHS ---
            st.info("Classification de la prévision par Poste Horosaisonnier (PHS)...")
            # Find the row for the selected site in the perimeter dataframe
            site_info_row = st.session_state.perimeter_df[st.session_state.perimeter_df['prm'] == selected_site]
            if not site_info_row.empty:
                site_info = site_info_row.iloc[0]

                # The helper function expects specific column names and 'Pas'
                forecast_for_phs = forecast_df.rename(columns={'datetime': 'Horodate', 'predicted_power_w': 'Valeur'})
                forecast_for_phs['Pas'] = f'{time_step_minutes} min'  # Format cohérent

                try:
                    periods, period_consumption, _ = get_time_periods(forecast_for_phs, site_info)
                    results['phs_consumption'] = period_consumption
                    results['classified_periods'] = periods
                    st.success("Classification PHS terminée avec succès.")

                    # Display PHS consumption in an expander
                    with st.expander("Consommation prévisionnelle par PHS (MWh)", expanded=True):
                        if period_consumption:
                            phs_df = pd.DataFrame.from_dict(period_consumption, orient='index', columns=['Consommation (MWh)'])
                            st.dataframe(phs_df)
                            fig_phs = px.pie(phs_df, values='Consommation (MWh)', names=phs_df.index, title='Répartition PHS')
                            st.plotly_chart(fig_phs, use_container_width=True)
                        else:
                            st.warning("Aucune consommation n'a pu être classifiée par PHS.")

                except Exception as e:
                    st.error(f"Erreur lors de la classification PHS : {e}")
                    st.exception(e)
            else:
                st.error(f"Impossible de trouver les informations du site {selected_site} dans le fichier périmètre.")

        st.session_state.forecast_results[selected_site] = results
        
        st.success(f"Prévision terminée pour {selected_site}!")
        st.success(f"MAE: {mae:.2f} W, R²: {r2:.3f}")
        st.info(f"Qualité des données: {final_count}/{initial_count} points utilisés ({(1-results['data_quality']['data_loss_ratio'])*100:.1f}% conservés)")
        st.info(f"Pas de temps original: {original_minutes:.0f} min, cible: {time_step_minutes} min")
        
        if forecast_df is not None:
            st.success(f"Prévision générée pour {len(forecast_df)} points (fréquence: {freq})")


def pricing_results_tab():
    """Onglet des résultats de pricing avec affichage dynamique."""
    st.header("📋 Résultats du Pricing")

    if not st.session_state.get('forecast_results') or st.session_state.get('price_curve_df') is None or st.session_state.price_curve_df.empty:
        st.warning("Veuillez d'abord effectuer les prévisions de consommation et charger une courbe de prix")
        return

    if not st.session_state.get('risk_premiums_config'):
        st.warning("Veuillez d'abord configurer les primes de risque dans l'onglet correspondant")
        return

    error_messages = []
    can_calculate = True
    price_curve_freq = None
    price_curve_df = st.session_state.get('price_curve_df')
    if price_curve_df is not None and len(price_curve_df) > 1:
        price_curve_timedelta = pd.to_datetime(price_curve_df['datetime'].iloc[1]) - pd.to_datetime(price_curve_df['datetime'].iloc[0])
        if price_curve_timedelta == timedelta(hours=1): price_curve_freq = "Horaire"
        elif price_curve_timedelta == timedelta(minutes=15): price_curve_freq = "15 minutes"

    forecast_freq = None
    forecast_results = st.session_state.get('forecast_results')
    if forecast_results:
        first_site = next(iter(forecast_results))
        forecast_freq = forecast_results[first_site].get('data_quality', {}).get('granularity')

    if price_curve_freq and forecast_freq and price_curve_freq != forecast_freq:
        error_messages.append(f"Incohérence de granularité ! La courbe de prix est au pas **{price_curve_freq}** tandis que la prévision est au pas **{forecast_freq}**.")

    quotation_info = st.session_state.quotation_info
    perimeter_df = st.session_state.get('perimeter_df')
    if perimeter_df is not None:
        global_start_date, global_end_date = pd.to_datetime(quotation_info['start_date']), pd.to_datetime(quotation_info['end_date'])
        overall_min_date, overall_max_date = global_start_date, global_end_date

        for _, site_row in perimeter_df.iterrows():
            site_details = st.session_state.site_details.get(str(site_row['prm']), {})
            site_start, site_end = pd.to_datetime(site_details.get('start_date', global_start_date)), pd.to_datetime(site_details.get('end_date', global_end_date))
            if not (global_start_date <= site_start and site_end <= global_end_date):
                error_messages.append(f"Site {site_row['prm']}: La période [{site_start.date()} - {site_end.date()}] n'est pas incluse dans la période de cotation globale.")
            overall_min_date, overall_max_date = min(overall_min_date, site_start), max(overall_max_date, site_end)

        if price_curve_df is not None:
            price_curve_start, price_curve_end = pd.to_datetime(price_curve_df['datetime'].min()), pd.to_datetime(price_curve_df['datetime'].max())
            if overall_min_date < price_curve_start or overall_max_date > price_curve_end:
                error_messages.append(f"La courbe de prix ne couvre pas toute la période de fourniture. Période requise : [{overall_min_date.date()} - {overall_max_date.date()}]. Période couverte : [{price_curve_start.date()} - {price_curve_end.date()}].")

        if forecast_results:
            for prm, forecast_res in forecast_results.items():
                site_details = st.session_state.site_details.get(str(prm), {})
                site_start, site_end = pd.to_datetime(site_details.get('start_date', global_start_date)).date(), pd.to_datetime(site_details.get('end_date', global_end_date)).date()
                if 'forecast_df' in forecast_res and not forecast_res['forecast_df'].empty:
                    forecast_start, forecast_end = forecast_res['forecast_df']['datetime'].min().date(), forecast_res['forecast_df']['datetime'].max().date()
                    if site_start != forecast_start or site_end != forecast_end:
                        error_messages.append(f"Site {prm}: La prévision a été générée pour la période [{forecast_start} - {forecast_end}], mais les dates du site sont maintenant [{site_start} - {site_end}]. Veuillez relancer la prévision.")

    if error_messages:
        can_calculate = False
        for msg in error_messages: st.error(msg)
    else: st.success("Toutes les vérifications de cohérence sont passées.")

    if st.button("🧮 Calculer le Pricing", disabled=not can_calculate):
        calculate_pricing_with_premiums()

    if st.session_state.pricing_results:
        st.subheader("Résumé Global")
        total_consumption = sum(r.get('total_consumption', 0) for r in st.session_state.pricing_results.values())
        total_cost_ttc = sum(pd.DataFrame(r.get('detailed_breakdown', [])).get('Coût Total TTC', 0).sum() for r in st.session_state.pricing_results.values())

        col_global1, col_global2, col_global3 = st.columns(3)
        with col_global1: st.metric("Sites Analysés", f"{len(st.session_state.pricing_results)}")
        with col_global2: st.metric("Énergie Totale", f"{total_consumption:.2f} MWh")
        with col_global3: st.metric("Coût Total TTC", f"{total_cost_ttc:,.0f} €")

        st.subheader("Résultats par Site")
        for site_id, results in st.session_state.pricing_results.items():
            with st.expander(f"Site {site_id}"):
                if not results.get('detailed_breakdown'):
                    st.warning("Aucun détail de pricing à afficher pour ce site.")
                    continue

                df = pd.DataFrame(results['detailed_breakdown'])

                # --- Indicateurs Principaux ---
                total_site_consumption = results['total_consumption']
                cout_total_htva = df['Coût Total HTVA'].sum()
                cout_total_ttc = df['Coût Total TTC'].sum()
                cout_acheminement = df['turpe_total_cost'].sum()
                cout_taxes = df['cta_total_cost'].sum() + df['excise_total_cost'].sum()
                # CORRECTION : inclure la C3S dans le coût énergie
                cout_energie = cout_total_htva - cout_acheminement - cout_taxes  # C3S déjà incluse
                cout_tva = cout_total_ttc - cout_total_htva

                st.markdown("##### Indicateurs Principaux (Total Période)")
                cols = st.columns(4)
                cols[0].metric("Energie Prévue", f"{total_site_consumption:.2f} MWh", help="Consommation totale d'énergie prévue sur la période de fourniture, basée sur les prévisions.")
                # MODIFICATION : préciser que la C3S est incluse
                cols[1].metric("Coût Energie", f"{cout_energie:,.0f} €", help="Coût de la fourniture d'énergie (prix de marché + primes + marge + capacité + GO + C3S).")
                cols[2].metric("Coût Acheminement", f"{cout_acheminement:,.0f} €", help="Coût du transport de l'électricité sur le réseau (TURPE), hors CTA.")
                cols[3].metric("Taxes & CTA", f"{cout_taxes:,.0f} €", help="Somme de la CTA (sur TURPE) et de l'Accise sur l'électricité.")
                cols = st.columns(3)
                cols[0].metric("Coût Total HTVA", f"{cout_total_htva:,.0f} €", help="Somme de tous les coûts hors TVA (Energie + Acheminement + Taxes + C3S).")
                cols[1].metric("Coût TVA", f"{cout_tva:,.0f} €", help="Montant de la Taxe sur la Valeur Ajoutée, calculée sur le total HTVA.")
                cols[2].metric("Coût Total TTC", f"{cout_total_ttc:,.0f} €", help="Coût final tout compris pour le client (HTVA + TVA).")

                # --- Tableau Détaillé ---
                is_yearly_view = df['Année'].dtype != 'object'
                title = "Détail par Année Civile" if is_yearly_view else "Synthèse de la Période"
                st.subheader(title)

                df_display = df.copy()
                if not is_yearly_view:
                    df_display = df_display.rename(columns={'Année': 'Période'})

                if 'PHS' not in df_display.columns:
                    df_display['PHS'] = 'Unique'

                rename_dict = {
                    "Coût Acheminement (€/MWh)": "Acheminement (€/MWh)", "Taxes (€/MWh)": "Taxes (€/MWh)",
                    "Prix total (€/MWh)": "Total (€/MWh)", "Coût Acheminement (€)": "Acheminement (€)",
                    "Taxes (€)": "Taxes (€)", "Prix total (€)": "Total (€)",
                    "Coût Total HTVA": "Total € HTVA", "Coût Total TTC": "Total € TTC"
                }
                df_display = df_display.rename(columns=rename_dict)

                col_order = ["Année", "Période", "PHS", "Consommation (MWh)", "Prix Client (€/MWh)", "Prix CEE (€/MWh)", "Acheminement (€/MWh)", "Taxes (€/MWh)", "Total (€/MWh)", "Acheminement (€)", "Taxes (€)", "Total (€)", "Total € HTVA", "Total € TTC"]
                display_cols = [col for col in col_order if col in df_display.columns]

                # Définir le format pour les colonnes numériques
                format_dict = {col: "{:,.2f}" for col in df_display.select_dtypes(include=np.number).columns}
                # Garder l'année comme entier
                if 'Année' in format_dict:
                    format_dict['Année'] = "{:d}"

                st.dataframe(df_display[display_cols].style.format(format_dict), use_container_width=True, hide_index=True)

                # --- Cost Breakdown and Pie Chart ---
                st.subheader("Détail des Coûts (Total Période)")

                # Construction dynamique du détail des coûts
                cost_breakdown = {'Énergie (base)': df['electron_base_cost'].sum()}

                # Détail des primes de risque
                if 'premiums_breakdown' in results and results['premiums_breakdown']:
                    for premium_name, premium_value in results['premiums_breakdown'].items():
                        # On multiplie la prime par MWh par la conso totale pour avoir le coût
                        cost_breakdown[f"Prime: {premium_name}"] = premium_value * total_site_consumption

                cost_breakdown['Marge Commerciale'] = df['marge_total_cost'].sum()
                cost_breakdown['Garanties d\'Origine'] = df['go_total_cost'].sum()
                cost_breakdown['Capacité'] = df['capacity_total_cost'].sum()
                cost_breakdown['CEE'] = df['cee_total_cost'].sum()

                # Détail de l'acheminement (TURPE)
                if 'turpe_breakdown' in results and results['turpe_breakdown']:
                    turpe_map = {
                        'composante_gestion': 'TURPE: Gestion',
                        'composante_comptage': 'TURPE: Comptage',
                        'composante_soutirage_puissance': 'TURPE: Puissance',
                        'composante_soutirage_energie': 'TURPE: Energie',
                        'penalites_depassement': 'TURPE: Dépassements',
                        'composante_energie_reactive': 'TURPE: Energie Réactive'
                    }
                    for key, name in turpe_map.items():
                        cost_breakdown[name] = results['turpe_breakdown'].get(key, 0)

                cost_breakdown['CTA'] = df['cta_total_cost'].sum()
                cost_breakdown['Accise'] = df['excise_total_cost'].sum()
                cost_breakdown['C3S'] = df['c3s_total_cost'].sum()
                cost_breakdown['TVA'] = df['vat_cost'].sum()

                cost_df = pd.DataFrame(cost_breakdown.items(), columns=['Composante', 'Coût Total (€)'])
                if total_site_consumption > 0:
                    cost_df['Coût (€/MWh)'] = cost_df['Coût Total (€)'] / total_site_consumption
                else:
                    cost_df['Coût (€/MWh)'] = 0

                cost_df = cost_df[['Composante', 'Coût (€/MWh)', 'Coût Total (€)']]
                st.dataframe(cost_df[cost_df['Coût Total (€)'].abs() > 0.01].style.format({'Coût (€/MWh)': '{:.3f}', 'Coût Total (€)': '{:,.2f} €'}), use_container_width=True, hide_index=True)

                # Agréger les coûts pour le graphique
                pie_chart_costs = {}
                pie_chart_costs['Primes de Risque'] = sum(v for k, v in cost_breakdown.items() if k.startswith('Prime:'))
                pie_chart_costs['Acheminement'] = sum(v for k, v in cost_breakdown.items() if k.startswith('TURPE:'))

                for k, v in cost_breakdown.items():
                    if not k.startswith('Prime:') and not k.startswith('TURPE:'):
                        pie_chart_costs[k] = v

                fig_pie = px.pie(values=[v for v in pie_chart_costs.values() if v > 0],
                                    names=[k for k, v in pie_chart_costs.items() if v > 0],
                                    title=f"Répartition des Coûts (Total Période) - Site {site_id}")
                st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_chart_{site_id}")

def calculate_pricing_with_premiums():
    """Calcule le pricing pour tous les sites avec une logique dynamique pour le type de prix."""
    with st.spinner("Calcul du pricing en cours..."):
        pricing_results = {}
        quotation_info = st.session_state.quotation_info
        tariff_params = st.session_state.tariff_params
        pricing_type = quotation_info.get('pricing_type', 'Prix lissé')
        offer_type = quotation_info.get('offer_type', 'Prix fixe unique')

        app_tariffs = st.session_state.app_tariffs
        turpe_calc = TURPECalculator(app_tariffs['TURPE'])
        cta_calc = CTACalculator(app_tariffs['Taxes']['cta_rate'])
        premium_calc = RiskPremiumCalculator()

        default_premiums = st.session_state.risk_premiums_config.get('default_premiums', {})
        premiums_df = st.session_state.get('premiums_df')

        marge_commerciale_eur_mwh = tariff_params.get('marge_commerciale', 5.0)
        green_option_activated = tariff_params.get('green_option_activated', False)
        cee_costs_yearly = tariff_params.get('cee_costs_yearly', {})
        go_costs_yearly = tariff_params.get('go_costs_yearly', {})
        capacity_costs_yearly = tariff_params.get('capacity_costs_yearly', {})
        c3s_rate = app_tariffs['Taxes'].get('c3s_rate', 0.0)
        excise_tax_eur_mwh = app_tariffs['Taxes'].get('excise_tax', 22.5)
        vat_rate = app_tariffs['Taxes'].get('vat_rate', 0.20)

        for site_id, forecast_result in st.session_state.forecast_results.items():
            price_df = st.session_state.price_curve_df.copy()
            site_info_row = st.session_state.perimeter_df[st.session_state.perimeter_df['prm'] == site_id]
            if site_info_row.empty: continue
            site_info = site_info_row.iloc[0]

            total_consumption_mwh = sum(forecast_result.get('phs_consumption', {}).values())
            if total_consumption_mwh <= 0: continue

            site_details = st.session_state.site_details.get(str(site_id), {})
            contract_start_date = pd.to_datetime(site_details.get('start_date', quotation_info['start_date']))
            contract_end_date = pd.to_datetime(site_details.get('end_date', quotation_info['end_date']))

            site_naf_code = site_details.get('naf_code', st.session_state.client_info.get('naf_code'))
            site_cee_eligible = 'Non'
            if site_naf_code and 'naf_codes' in st.session_state and not st.session_state.naf_codes.empty:
                naf_info = st.session_state.naf_codes[st.session_state.naf_codes['Code NAF'] == site_naf_code]
                if not naf_info.empty: site_cee_eligible = naf_info.iloc[0]['Eligible CEE']

            classified_periods = forecast_result.get('classified_periods', {})
            if not classified_periods: continue

            temp_dfs = [df.copy().assign(PHS=period_name) for period_name, df in classified_periods.items()]
            forecast_with_phs_df = pd.concat(temp_dfs).sort_values('datetime').rename(columns={'Valeur': 'predicted_power_w'})

            site_premiums = premium_calc.get_site_premiums(site_id, premiums_df, default_premiums)
            total_premium_eur_mwh = premium_calc.calculate_total_premium(site_premiums)

            merged_df = pd.merge(forecast_with_phs_df, price_df, on='datetime', how='inner')
            if merged_df.empty: continue

            # CORRECTION MAJEURE : Calcul correct de l'énergie MWh
            # Récupérer le pas de temps original depuis les données de forecast
            original_step_minutes = forecast_result.get('data_quality', {}).get('original_step_minutes', 5)
            forecast_step_minutes = forecast_result.get('data_quality', {}).get('time_step_minutes', 60)
            
            st.info(f"Site {site_id}: Pas original {original_step_minutes}min, pas forecast {forecast_step_minutes}min")
            
            # predicted_power_w est une puissance moyenne sur l'intervalle de forecast
            # Pour obtenir l'énergie en MWh sur cet intervalle :
            merged_df['energy_mwh'] = (merged_df['predicted_power_w'] * forecast_step_minutes / 60) / 1_000_000
            
            merged_df['year'] = pd.to_datetime(merged_df['datetime']).dt.year

            # Define grouping keys based on pricing and offer type
            grouping_keys = []
            if pricing_type == 'Prix par année': grouping_keys.append('year')
            if offer_type == 'Prix fixe par PHS': grouping_keys.append('PHS')

            if not grouping_keys:
                grouped = [('Globale', merged_df)]
            else:
                grouped = merged_df.groupby(grouping_keys)

            # BUG FIX: Calculate total TURPE breakdown for the site ONCE before the loop
            site_turpe_details = turpe_calc.calculate_current_cost(
                site_info,
                merged_df.rename(columns={'predicted_power_w': 'Valeur'}),
                start_date=contract_start_date,
                end_date=contract_end_date
            )
            if not site_turpe_details:
                site_turpe_details = {
                    'composante_gestion': 0, 'composante_comptage': 0, 'composante_soutirage_puissance': 0,
                    'composante_soutirage_energie': 0, 'penalites_depassement': 0, 'composante_energie_reactive': 0
                }

            detailed_breakdown = []
            for group_key, group_df in grouped:
                year, phs = None, 'Unique'
                if isinstance(group_key, tuple):
                    key_iter = iter(group_key)
                    if 'year' in grouping_keys:
                        year = next(key_iter, None)
                    if 'PHS' in grouping_keys:
                        phs = next(key_iter, None)
                else:
                    if 'year' in grouping_keys:
                        year = group_key
                    if 'PHS' in grouping_keys:
                        phs = group_key

                group_consumption_mwh = group_df['energy_mwh'].sum()
                if group_consumption_mwh <= 0: continue

                # Determine costs for the relevant period (year or total)
                period_df = merged_df[merged_df['year'] == year] if year else merged_df
                period_consumption_mwh = period_df['energy_mwh'].sum()

                period_start = max(contract_start_date, pd.Timestamp(year, 1, 1)) if year else contract_start_date
                period_end = min(contract_end_date, pd.Timestamp(year, 12, 31)) if year else contract_end_date

                # BUG FIX: Calculate TURPE details for the period (not the whole site) to get the fixed part
                turpe_details_for_period_calc = turpe_calc.calculate_current_cost(site_info, period_df.rename(columns={'predicted_power_w': 'Valeur'}), start_date=period_start, end_date=period_end)

                turpe_fixed_for_period = turpe_details_for_period_calc.get('composante_gestion', 0) + turpe_details_for_period_calc.get('composante_comptage', 0) + turpe_details_for_period_calc.get('composante_soutirage_puissance', 0)

                cta_details_period = cta_calc.calculate_cta(turpe_details_for_period_calc)
                cta_fixed_period = cta_details_period.get('total_cta', 0)

                # Allocate fixed costs based on consumption share
                consumption_share = group_consumption_mwh / period_consumption_mwh if period_consumption_mwh > 0 else 0

                # Per-group costs
                cee_cost_eur_mwh = cee_costs_yearly.get(str(year), 12.0) if year and site_cee_eligible == 'Oui' else (sum(cee_costs_yearly.values())/len(cee_costs_yearly) if cee_costs_yearly and site_cee_eligible == 'Oui' else 0)
                go_cost_eur_mwh = go_costs_yearly.get(str(year), 0.8) if year and green_option_activated else (sum(go_costs_yearly.values())/len(go_costs_yearly) if go_costs_yearly and green_option_activated else 0)
                capacity_cost_eur_mwh = capacity_costs_yearly.get(str(year), 3.0) if year else (sum(capacity_costs_yearly.values())/len(capacity_costs_yearly) if capacity_costs_yearly else 0)

                electron_base_cost = (group_df['energy_mwh'] * group_df['prix_euro_mwh']).sum()
                premiums_total_cost = group_consumption_mwh * total_premium_eur_mwh
                marge_total_cost = group_consumption_mwh * marge_commerciale_eur_mwh
                cee_total_cost = group_consumption_mwh * cee_cost_eur_mwh
                go_total_cost = group_consumption_mwh * go_cost_eur_mwh
                capacity_total_cost = group_consumption_mwh * capacity_cost_eur_mwh

                raw_formula = site_info.get("Formule tarifaire acheminement")
                current_formula = turpe_calc.map_formula_to_internal(raw_formula)
                group_periods, _, time_step_hours = get_time_periods(group_df.rename(columns={'predicted_power_w': 'Valeur'}), site_info)
                turpe_variable_cost = turpe_calc.calculate_energy_cost(group_periods, site_info, current_formula, time_step_hours)
                turpe_fixed_cost = turpe_fixed_for_period * consumption_share
                turpe_total_cost = turpe_fixed_cost + turpe_variable_cost

                cta_total_cost = cta_fixed_period * consumption_share
                excise_total_cost = group_consumption_mwh * excise_tax_eur_mwh

                cout_energie_before_c3s = electron_base_cost + premiums_total_cost + marge_total_cost + cee_total_cost + go_total_cost + capacity_total_cost
                total_cost_ht_before_c3s = cout_energie_before_c3s + turpe_total_cost + cta_total_cost + excise_total_cost
                c3s_total_cost = total_cost_ht_before_c3s * c3s_rate

                total_cost_ht = total_cost_ht_before_c3s + c3s_total_cost
                vat_cost = total_cost_ht * vat_rate

                prix_client_total_cost = electron_base_cost + premiums_total_cost + marge_total_cost + capacity_total_cost + c3s_total_cost + go_total_cost
                prix_client_eur_mwh = prix_client_total_cost / group_consumption_mwh if group_consumption_mwh > 0 else 0

                row = {
                    'Année': year if year else "Globale", 'PHS': phs, 'Consommation (MWh)': group_consumption_mwh,
                    'Prix Client (€/MWh)': prix_client_eur_mwh, 'Prix CEE (€/MWh)': cee_cost_eur_mwh,
                    'Coût Acheminement (€)': turpe_fixed_cost,
                    'Coût Acheminement (€/MWh)': turpe_variable_cost / group_consumption_mwh if group_consumption_mwh > 0 else 0,
                    'Taxes (€)': cta_total_cost, 'Taxes (€/MWh)': excise_tax_eur_mwh,
                    'Prix total (€)': turpe_fixed_cost + cta_total_cost,
                    'Prix total (€/MWh)': prix_client_eur_mwh + cee_cost_eur_mwh + (turpe_variable_cost / group_consumption_mwh if group_consumption_mwh > 0 else 0) + excise_tax_eur_mwh,
                    'Coût Total HTVA': total_cost_ht, 'Coût Total TTC': total_cost_ht + vat_cost,
                    # Granular costs for aggregation in display tab
                    'electron_base_cost': electron_base_cost,
                    'premiums_total_cost': premiums_total_cost,
                    'marge_total_cost': marge_total_cost,
                    'cee_total_cost': cee_total_cost,
                    'go_total_cost': go_total_cost,
                    'capacity_total_cost': capacity_total_cost,
                    'c3s_total_cost': c3s_total_cost,
                    'turpe_total_cost': turpe_total_cost,
                    'cta_total_cost': cta_total_cost,
                    'excise_total_cost': excise_total_cost,
                    'vat_cost': vat_cost,
                }
                detailed_breakdown.append(row)

            # Aggregate totals for the site
            df_breakdown = pd.DataFrame(detailed_breakdown)
            total_consumption_site = df_breakdown['Consommation (MWh)'].sum()

            pricing_results[site_id] = {
                'total_consumption': total_consumption_site,
                'premiums_breakdown': site_premiums,
                'turpe_breakdown': site_turpe_details,
                'detailed_breakdown': detailed_breakdown,
                'merged_data': merged_df,
                'site_naf_code': site_naf_code,
                'site_cee_eligible': site_cee_eligible,
            }
        st.session_state.pricing_results = pricing_results
        st.success(f"Pricing calculé pour {len(pricing_results)} site(s)!")

def export_tab():
    """Onglet d'export des résultats avec la nouvelle structure."""
    st.header("📤 Export des Résultats")
    
    if not st.session_state.get('pricing_results'):
        st.warning("Aucun résultat de pricing à exporter. Veuillez d'abord lancer un calcul dans l'onglet 'Résultats Pricing'.")
        return
    
    st.subheader("Options d'Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Format d'export",
            ["Excel (.xlsx)", "ZIP (JSON + Prévisions CSV)"]
        )
        include_forecast_csv = st.checkbox("Inclure les prévisions CSV détaillées par site", value=True)

    with col2:
        st.subheader("Aperçu des Données à Exporter")
        total_sites = len(st.session_state.pricing_results)
        total_consumption = sum(r.get('total_consumption', 0) for r in st.session_state.pricing_results.values())
        total_cost_ttc = sum(pd.DataFrame(r.get('detailed_breakdown', [])).get('Coût Total TTC', 0).sum() for r in st.session_state.pricing_results.values())

        st.metric("Nombre de Sites", total_sites)
        st.metric("Énergie Totale", f"{total_consumption:.2f} MWh")
        st.metric("Coût Total TTC", f"{total_cost_ttc:,.2f} €")
    
    if st.button("📥 Générer l'Export"):
        generate_export(export_format, include_forecast_csv)


def generate_export(export_format, include_forecast_csv):
    """
    Génère les fichiers d'export (Excel ou ZIP) selon la nouvelle structure demandée.
    """
    with st.spinner("Génération de l'export en cours..."):
        if export_format == "Excel (.xlsx)":
            excel_buffer = _generate_excel_export(include_forecast_csv)
            st.download_button(
                label="📥 Télécharger l'Export Excel",
                data=excel_buffer.getvalue(),
                file_name=f"cotation_{st.session_state.quotation_info.get('quotation_id', 'export')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        elif export_format == "ZIP (JSON + Prévisions CSV)":
            zip_buffer = _generate_zip_export(include_forecast_csv)
            st.download_button(
                label="📥 Télécharger l'Export ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"cotation_{st.session_state.quotation_info.get('quotation_id', 'export')}.zip",
                mime="application/zip"
            )
        st.success("Export généré avec succès !")

# --- Fonctions de préparation des données pour l'export ---

def _prepare_resume_df():
    """Prépare le DataFrame pour l'onglet 'Résumé' de l'export Excel."""
    summary_data = []
    quotation_id = st.session_state.quotation_info.get('quotation_id')
    global_start_date = st.session_state.quotation_info.get('start_date')
    global_end_date = st.session_state.quotation_info.get('end_date')

    for site_id, results in st.session_state.pricing_results.items():
        if not results.get('detailed_breakdown'):
            continue

        df = pd.DataFrame(results['detailed_breakdown'])
        site_details = st.session_state.site_details.get(str(site_id), {})
        start_date = site_details.get('start_date', global_start_date)
        end_date = site_details.get('end_date', global_end_date)

        total_consumption = results.get('total_consumption', 0)
        cout_total_htva = df['Coût Total HTVA'].sum()
        cout_total_ttc = df['Coût Total TTC'].sum()
        cout_acheminement = df['turpe_total_cost'].sum()
        cout_taxes_cta = df['cta_total_cost'].sum() + df['excise_total_cost'].sum()
        cout_tva = df['vat_cost'].sum()
        
        # CORRECTION : Le coût de l'énergie inclut maintenant la C3S
        # (pas de changement car c3s_total_cost est déjà inclus dans cout_total_htva)
        cout_energie = cout_total_htva - cout_acheminement - cout_taxes_cta

        summary_data.append({
            'site_id': site_id,
            'id_cotation': quotation_id,
            'date_debut_fourniture': start_date,
            'date_fin_fourniture': end_date,
            'Energie Prevue (MWh)': total_consumption,
            'Coût Energie (€)': cout_energie,  # C3S maintenant incluse
            'Coût Acheminement (€)': cout_acheminement,
            'Taxes et CTA (€)': cout_taxes_cta,
            'Coût Total HTVA (€)': cout_total_htva,
            'Coût TVA (€)': cout_tva,
            'Coût Total TTC (€)': cout_total_ttc,
        })

    return pd.DataFrame(summary_data)

def _prepare_detail_cotation_df():
    """Prépare le DataFrame pour l'onglet 'Détail Cotation' de l'export Excel."""
    all_sites_cost_details = []
    quotation_id = st.session_state.quotation_info.get('quotation_id')
    global_start_date = st.session_state.quotation_info.get('start_date')
    global_end_date = st.session_state.quotation_info.get('end_date')

    for site_id, results in st.session_state.pricing_results.items():
        if not results.get('detailed_breakdown'):
            continue

        df_breakdown = pd.DataFrame(results['detailed_breakdown'])
        total_site_consumption = results.get('total_consumption', 0)

        site_details = st.session_state.site_details.get(str(site_id), {})
        start_date = site_details.get('start_date', global_start_date)
        end_date = site_details.get('end_date', global_end_date)

        cost_breakdown = {'Énergie (base)': df_breakdown['electron_base_cost'].sum()}
        if 'premiums_breakdown' in results and results['premiums_breakdown']:
            for name, value in results['premiums_breakdown'].items():
                cost_breakdown[f"Prime: {name}"] = value * total_site_consumption

        cost_breakdown['Marge Commerciale'] = df_breakdown['marge_total_cost'].sum()
        cost_breakdown['Garanties d\'Origine'] = df_breakdown['go_total_cost'].sum()
        cost_breakdown['Capacité'] = df_breakdown['capacity_total_cost'].sum()
        cost_breakdown['CEE'] = df_breakdown['cee_total_cost'].sum()

        if 'turpe_breakdown' in results and results['turpe_breakdown']:
            turpe_map = {
                'composante_gestion': 'TURPE: Gestion', 'composante_comptage': 'TURPE: Comptage',
                'composante_soutirage_puissance': 'TURPE: Puissance', 'composante_soutirage_energie': 'TURPE: Energie',
                'penalites_depassement': 'TURPE: Dépassements', 'composante_energie_reactive': 'TURPE: Energie Réactive'
            }
            for key, name in turpe_map.items():
                cost_breakdown[name] = results['turpe_breakdown'].get(key, 0)

        cost_breakdown['CTA'] = df_breakdown['cta_total_cost'].sum()
        cost_breakdown['Accise'] = df_breakdown['excise_total_cost'].sum()
        cost_breakdown['C3S'] = df_breakdown['c3s_total_cost'].sum()
        cost_breakdown['TVA'] = df_breakdown['vat_cost'].sum()

        cost_df = pd.DataFrame(cost_breakdown.items(), columns=['Composante', 'Coût Total (€)'])
        cost_df['Coût (€/MWh)'] = cost_df['Coût Total (€)'] / total_site_consumption if total_site_consumption > 0 else 0

        cost_df['site_id'] = site_id
        cost_df['id_cotation'] = quotation_id
        cost_df['date_debut_fourniture'] = start_date
        cost_df['date_fin_fourniture'] = end_date

        all_sites_cost_details.append(cost_df)

    if not all_sites_cost_details:
        return pd.DataFrame()

    final_df = pd.concat(all_sites_cost_details, ignore_index=True)
    # Réorganiser les colonnes
    column_order = ['site_id', 'id_cotation', 'date_debut_fourniture', 'date_fin_fourniture', 'Composante', 'Coût (€/MWh)', 'Coût Total (€)']
    return final_df[column_order]


def _prepare_prix_final_df():
    """Prépare le DataFrame pour l'onglet 'Prix Final' de l'export Excel."""
    all_sites_details = []
    quotation_id = st.session_state.quotation_info.get('quotation_id')
    global_start_date_dt = pd.to_datetime(st.session_state.quotation_info.get('start_date'))
    global_end_date_dt = pd.to_datetime(st.session_state.quotation_info.get('end_date'))

    for site_id, results in st.session_state.pricing_results.items():
        if 'detailed_breakdown' in results and results['detailed_breakdown']:
            df = pd.DataFrame(results['detailed_breakdown'])
            
            site_details = st.session_state.site_details.get(str(site_id), {})
            contract_start = pd.to_datetime(site_details.get('start_date', global_start_date_dt))
            contract_end = pd.to_datetime(site_details.get('end_date', global_end_date_dt))

            df['site_id'] = site_id
            df['id_cotation'] = quotation_id

            # Calculer les dates de début et de fin pour chaque ligne
            dates = []
            for _, row in df.iterrows():
                if row['Année'] == 'Globale':
                    dates.append({'date_debut': contract_start.date(), 'date_fin': contract_end.date()})
                else:
                    year = int(row['Année'])
                    start_of_year = pd.Timestamp(year, 1, 1)
                    end_of_year = pd.Timestamp(year, 12, 31)
                    period_start = max(contract_start, start_of_year)
                    period_end = min(contract_end, end_of_year)
                    dates.append({'date_debut': period_start.date(), 'date_fin': period_end.date()})

            dates_df = pd.DataFrame(dates)
            df_with_dates = pd.concat([df.reset_index(drop=True), dates_df.reset_index(drop=True)], axis=1)
            all_sites_details.append(df_with_dates)

    if not all_sites_details:
        return pd.DataFrame()

    return pd.concat(all_sites_details, ignore_index=True)


def _prepare_forecast_df(site_id, results):
    """Prépare le DataFrame de prévision pour un site donné."""
    if 'merged_data' in results and not results['merged_data'].empty:
        df_to_export = results['merged_data']
        columns_to_keep = {
            'datetime': 'Date_Heure',
            'PHS': 'PHS',
            'energy_mwh': 'Energie_MWh',
            'prix_euro_mwh': 'Prix_Marche_EUR_MWh'
        }
        # Filtrer les colonnes qui existent réellement
        existing_columns = {k: v for k, v in columns_to_keep.items() if k in df_to_export.columns}
        df_filtered = df_to_export[list(existing_columns.keys())].copy()
        df_filtered = df_filtered.rename(columns=existing_columns)
        return df_filtered
    return pd.DataFrame()

# --- Fonctions de génération des fichiers ---

def _generate_excel_export(include_forecast_csv):
    """Génère le contenu du fichier Excel dans un buffer."""
    from io import BytesIO
    buffer = BytesIO()

    def normalize_df_for_export(df: pd.DataFrame) -> pd.DataFrame:
        """Applique la normalisation sur les colonnes et le contenu d'un DataFrame."""
        if df.empty:
            return df

        df_copy = df.copy()
        df_copy.columns = [normalize_text(col) for col in df_copy.columns]
        for col in df_copy.select_dtypes(include=['object']).columns:
            df_copy[col] = df_copy[col].apply(normalize_text)
        return df_copy

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Onglets non normalisés (données brutes)
        client_info_df = pd.DataFrame([st.session_state.client_info])
        normalize_df_for_export(client_info_df).to_excel(writer, sheet_name='Client', index=False)

        quotation_info_export = st.session_state.quotation_info.copy()
        if 'quotation_time' in quotation_info_export and hasattr(quotation_info_export.get('quotation_time'), 'strftime'):
            quotation_info_export['quotation_time'] = quotation_info_export['quotation_time'].strftime('%H:%M:%S')

        cotation_df = pd.DataFrame([quotation_info_export])
        normalize_df_for_export(cotation_df).to_excel(writer, sheet_name='Cotation', index=False)

        if st.session_state.get('perimeter_df') is not None:
            perimeter_df = st.session_state.perimeter_df
            normalize_df_for_export(perimeter_df).to_excel(writer, sheet_name='Caracteristiques_Sites', index=False)

        # Nouveaux onglets avec normalisation
        resume_df = _prepare_resume_df()
        normalize_df_for_export(resume_df).to_excel(writer, sheet_name='Resume', index=False)

        detail_cotation_df = _prepare_detail_cotation_df()
        normalize_df_for_export(detail_cotation_df).to_excel(writer, sheet_name='Detail_Cotation', index=False)

        prix_final_df = _prepare_prix_final_df()
        normalize_df_for_export(prix_final_df).to_excel(writer, sheet_name='Prix_final', index=False)

        # Onglet prévision (si demandé)
        if include_forecast_csv:
            for site_id, results in st.session_state.pricing_results.items():
                forecast_df = _prepare_forecast_df(site_id, results)
                if not forecast_df.empty:
                    # Le nom de l'onglet Excel est limité à 31 caractères
                    sheet_name = normalize_text(f'Prevision_{site_id}')[:31]
                    normalize_df_for_export(forecast_df).to_excel(writer, sheet_name=sheet_name, index=False)

    buffer.seek(0)
    return buffer

def _generate_zip_export(include_forecast_csv):
    """Génère le contenu du fichier ZIP dans un buffer avec la nouvelle structure JSON."""
    zip_buffer = io.BytesIO()

    # --- 1. Préparation de la structure de données JSON ---
    quotation_info = st.session_state.quotation_info.copy()

    # Résumé global
    total_sites = len(st.session_state.pricing_results)
    total_consumption = sum(r.get('total_consumption', 0) for r in st.session_state.pricing_results.values())
    total_cost_ttc = sum(pd.DataFrame(r.get('detailed_breakdown', [])).get('Coût Total TTC', 0).sum() for r in st.session_state.pricing_results.values())

    export_data = {
        "info_cotation": quotation_info,
        "info_client": st.session_state.client_info,
        "parametres_offre": {
            "type_offre": quotation_info.get('offer_type'),
            "type_calcul_prix": quotation_info.get('pricing_type')
        },
        "resume_global": {
            "nombre_sites": total_sites,
            "energie_totale_mwh": total_consumption,
            "cout_total_ttc_eur": total_cost_ttc
        },
        "sites": []
    }

    # Données par site
    resume_df = _prepare_resume_df()
    detail_cotation_df = _prepare_detail_cotation_df()
    prix_final_df = _prepare_prix_final_df()

    for site_id, results in st.session_state.pricing_results.items():
        site_resume_list = resume_df[resume_df['site_id'] == site_id].to_dict('records')
        site_resume = site_resume_list[0] if site_resume_list else {}
        
        site_detail_costs = detail_cotation_df[detail_cotation_df['site_id'] == site_id][['Composante', 'Coût (€/MWh)', 'Coût Total (€)']].to_dict('records')
        site_prix_final = prix_final_df[prix_final_df['site_id'] == site_id].drop(columns=['site_id', 'id_cotation']).to_dict('records')

        site_data = {
            "info_site": {
                "site_id": site_id,
                "code_naf": results.get('site_naf_code', 'N/A'),
                "eligible_cee": results.get('site_cee_eligible', 'Non'),
                "date_debut_fourniture": site_resume.get('date_debut_fourniture'),
                "date_fin_fourniture": site_resume.get('date_fin_fourniture'),
            },
            "resume_site": site_resume,
            "detail_couts_unitaire": site_detail_costs,
            "prix_final": site_prix_final
        }
        export_data['sites'].append(site_data)

    # --- 2. Sérialisation et écriture dans le ZIP ---

    # Helper pour convertir les types non sérialisables
    def default_converter(o):
        if isinstance(o, (datetime, pd.Timestamp, date)):
            return o.isoformat()
        if isinstance(o, time):
            return o.strftime('%H:%M:%S')
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if pd.isna(o):
            return None
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable: {o}")

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Normalisation de l'objet JSON avant l'export
        normalized_export_data = normalize_json_object(export_data)
        json_data = json.dumps(normalized_export_data, indent=4, default=default_converter)
        zip_file.writestr("cotation.json", json_data)

        if include_forecast_csv:
            for site_id, results in st.session_state.pricing_results.items():
                forecast_df = _prepare_forecast_df(site_id, results)
                if not forecast_df.empty:
                    csv_data = forecast_df.to_csv(index=False, sep=';', decimal=',')
                    zip_file.writestr(f"prevision_{site_id}.csv", csv_data)

    zip_buffer.seek(0)
    return zip_buffer

if __name__ == "__main__":
    main()