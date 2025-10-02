import pandas as pd
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

def detect_tension_level(site_data: pd.Series) -> str:
    """Détecte le niveau de tension basé sur les caractéristiques du site."""
    tension_str = site_data.get('Domaine de tension', '').upper()

    # Priorité 1: Le champ "Domaine de tension"
    if 'HTA' in tension_str:
        return 'HTA'
    if 'BTSUP' in tension_str:
        return 'BT_SUP36'
    if 'BTINF' in tension_str:
        return 'BT_INF36'

    # Priorité 2: La puissance (fallback)
    puissance_max_raw = site_data.get('Puissance limite soutirage', '0')
    puissance_max = 0
    if isinstance(puissance_max_raw, str):
        # Utiliser regex pour extraire le nombre, gère les formats comme "250.0 kVA"
        match = re.search(r'(\d+[.,]?\d*)', puissance_max_raw)
        if match:
            try:
                puissance_max = float(match.group(1).replace(',', '.'))
            except ValueError:
                puissance_max = 0
    elif isinstance(puissance_max_raw, (int, float)):
        puissance_max = puissance_max_raw

    if puissance_max > 1000:
        return 'HTA'
    elif puissance_max > 36:
        return 'BT_SUP36'
    else:
        return 'BT_INF36'

def parse_time_ranges(site_info: pd.Series) -> Dict[str, list]:
    """Parse 'poste horaire' fields from site_info to extract time ranges."""
    time_ranges = {'P': [], 'HC': []}
    # Regex to find "Poste horaire X" or "poste horaire X" case-insensitively
    for col in site_info.index:
        if re.match(r'poste horaire \d', col, re.IGNORECASE):
            value = site_info[col]
            if pd.notna(value) and isinstance(value, str):
                parts = value.split(':')
                if len(parts) > 1:
                    period_type = parts[0].strip().upper()
                    if period_type in ['P', 'HC']:
                        time_def = ':'.join(parts[1:]).strip()
                        ranges = time_def.split(';')
                        for r in ranges:
                            try:
                                start_str, end_str = [t.strip() for t in r.split('-')]
                                start_hour = int(start_str.split('h')[0])
                                end_hour = int(end_str.split('h')[0])

                                if end_hour == 0: end_hour = 24

                                if start_hour < end_hour:
                                    time_ranges[period_type].append((start_hour, end_hour))
                                else:  # Wraps around midnight
                                    time_ranges[period_type].append((start_hour, 24))
                                    time_ranges[period_type].append((0, end_hour))
                            except (ValueError, IndexError):
                                # Ignore malformed ranges
                                pass
    return time_ranges

def get_time_periods(consumption_data: pd.DataFrame, site_info: pd.Series) -> Tuple[Dict[str, pd.DataFrame], Dict[str, float], float]:
    """
    Classifie les données de consommation par périodes horaires et calcule la consommation totale pour chaque période.
    Retourne un tuple: (dictionnaire des périodes, dictionnaire des consommations par période en MWh, pas de temps en heures).
    """
    if 'Horodate' not in consumption_data.columns or 'Valeur' not in consumption_data.columns:
        raise ValueError("Le DataFrame de consommation doit contenir les colonnes 'Horodate' et 'Valeur'.")

    # Make a copy to avoid SettingWithCopyWarning
    consumption_data = consumption_data.copy()

    consumption_data['datetime'] = pd.to_datetime(consumption_data['Horodate'])
    consumption_data['hour'] = consumption_data['datetime'].dt.hour
    consumption_data['month'] = consumption_data['datetime'].dt.month
    consumption_data['dow'] = consumption_data['datetime'].dt.dayofweek  # 0=Lundi, 6=Dimanche

    # Définition des saisons (Saison Haute: Nov-Mars)
    consumption_data['saison'] = consumption_data['month'].apply(
        lambda x: 'HAUTE' if x in [11, 12, 1, 2, 3] else 'BASSE'
    )

    # Extraction des plages horaires depuis les données du site
    time_ranges = parse_time_ranges(site_info)
    peak_ranges = time_ranges.get('P', [])
    offpeak_ranges = time_ranges.get('HC', [])

    # Définition des heures pleines/creuses
    def classify_hour(row: pd.Series) -> str:
        hour = row['hour']
        month = row['month']
        dow = row['dow']
        saison = row['saison']

        # Sundays are always off-peak
        if dow == 6: # 6 is Sunday
            return 'HCH' if saison == 'HAUTE' else 'HCB'

        # Heures de pointe: Dec-Fev, Lun-Sam
        is_peak_season = month in [12, 1, 2]
        is_peak_day = dow < 6  # Lundi à Samedi

        if is_peak_season and is_peak_day:
            # Check for peak hours
            is_peak_hour = any(start <= hour < end for start, end in peak_ranges)
            if is_peak_hour:
                return 'POINTE'

        # Off-peak hours
        is_offpeak_hour = any(start <= hour < end for start, end in offpeak_ranges)
        if is_offpeak_hour:
            return 'HCH' if saison == 'HAUTE' else 'HCB'

        # Full hours (everything else)
        return 'HPH' if saison == 'HAUTE' else 'HPB'

    consumption_data['periode'] = consumption_data.apply(classify_hour, axis=1)

    # Déterminer le pas de temps
    time_step_hours = 0.5 # Default value
    if 'Pas' in consumption_data.columns and pd.notna(consumption_data['Pas'].iloc[0]):
        pas_str = consumption_data['Pas'].iloc[0]
        minutes_match = re.search(r'(\d+)', pas_str)
        if minutes_match:
            minutes = int(minutes_match.group(1))
            time_step_hours = minutes / 60.0
    elif not consumption_data['datetime'].empty:
        # Fallback si la colonne "Pas" n'est pas dispo
        time_diff = consumption_data['datetime'].diff().mode()
        if not time_diff.empty and pd.notna(time_diff[0]):
            time_step_hours = time_diff[0].total_seconds() / 3600

    # Calcul de l'énergie pour chaque enregistrement en Wh
    consumption_data['energie_wh'] = consumption_data['Valeur'] * time_step_hours

    # Groupement par périodes
    periods = {str(period): data for period, data in consumption_data.groupby('periode')}

    # Calcul de la consommation totale par période en MWh
    period_consumption_kwh = consumption_data.groupby('periode')['energie_wh'].sum() / 1_000_000

    return periods, period_consumption_kwh.to_dict(), time_step_hours

def get_segment(tension_level: str, formula: str) -> str:
    """Détermine le segment de marché (C2-C5) à partir du niveau de tension et de la formule tarifaire."""
    formula_upper = str(formula).upper()

    if tension_level == 'HTA':
        if 'LU' in formula_upper:
            return 'C2'
        elif 'CU' in formula_upper:
            return 'C3'
        else:
            return 'HTA (Formule non reconnue)'
    elif tension_level == 'BT_SUP36':
        return 'C4'
    elif tension_level == 'BT_INF36':
        return 'C5'

    return 'Indéterminé'

def calculate_business_hours(start_dt: datetime, end_dt: datetime, business_hours: Tuple[int, int] = (9, 17)) -> float:
    """
    Calcule le nombre total d'heures ouvrées entre deux datetimes.
    Les heures ouvrées sont définies du lundi au vendredi, de 9h à 17h.
    """
    if start_dt >= end_dt:
        return 0.0

    total_hours = 0.0
    current_dt = start_dt

    # Heures de travail par jour
    business_start_hour, business_end_hour = business_hours

    # Itérer jour par jour
    while current_dt.date() <= end_dt.date():
        # Ne compter que les jours de semaine (lundi=0, dimanche=6)
        if current_dt.weekday() < 5:
            # Définir le début et la fin de la journée de travail pour le jour actuel
            day_start_business = current_dt.replace(hour=business_start_hour, minute=0, second=0, microsecond=0)
            day_end_business = current_dt.replace(hour=business_end_hour, minute=0, second=0, microsecond=0)

            # Déterminer la période d'intersection pour le calcul
            # Le début de la période de calcul est le plus tardif entre le début de l'offre et le début des heures ouvrées ce jour-là
            start_calc = max(current_dt, day_start_business)
            # La fin de la période de calcul est le plus tôt entre la fin de l'offre et la fin des heures ouvrées ce jour-là
            end_calc = min(end_dt, day_end_business)

            # Si la période de calcul est valide (le début est avant la fin), calculer la durée
            if start_calc < end_calc:
                duration = end_calc - start_calc
                total_hours += duration.total_seconds() / 3600

        # Passer au jour suivant à minuit
        current_dt = (current_dt + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    return total_hours

def normalize_text(text: str) -> str:
    """
    Remplace les caractères spéciaux par leurs équivalents ASCII.
    Gère également les cas où l'entrée n'est pas une chaîne de caractères.
    """
    if not isinstance(text, str):
        return text

    replacements = {
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'à': 'a', 'â': 'a',
        'ô': 'o',
        'ù': 'u', 'û': 'u', 'ü': 'u',
        'î': 'i', 'ï': 'i',
        'ç': 'c',
        'É': 'E', 'È': 'E', 'Ê': 'E', 'Ë': 'E',
        'À': 'A', 'Â': 'A',
        'Ô': 'O',
        'Ù': 'U', 'Û': 'U', 'Ü': 'U',
        'Î': 'I', 'Ï': 'I',
        'Ç': 'C',
        '€': 'Eur',
        ' ': '_', # Remplacer les espaces par des underscores pour les noms de champs
    }

    for special, standard in replacements.items():
        text = text.replace(special, standard)

    return text

def normalize_json_object(obj: Any) -> Any:
    """
    Parcourt récursivement un objet (dictionnaire ou liste) et normalise
    toutes les clés de dictionnaire et toutes les valeurs de type chaîne de caractères.
    """
    if isinstance(obj, dict):
        return {normalize_text(k): normalize_json_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [normalize_json_object(elem) for elem in obj]
    elif isinstance(obj, str):
        return normalize_text(obj)
    else:
        return obj
