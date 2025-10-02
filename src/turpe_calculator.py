import pandas as pd
import numpy as np
import re
import streamlit as st
from typing import Dict, Optional

from src.helpers import get_time_periods, detect_tension_level

class TURPECalculator:
    """
    Classe pour le calcul du coût d'acheminement (TURPE) basé sur la configuration actuelle du site.
    """

    def __init__(self, tariffs: Dict):
        self.tariffs = tariffs

    def get_current_powers(self, site_data: pd.Series) -> Dict[str, float]:
        """Extrait les puissances souscrites actuelles à partir des données du site."""
        period_mapping = {
            'PTE': 'POINTE', 'POINTE': 'POINTE',
            'HPH': 'HPH',
            'HCH': 'HCH',
            'HPB': 'HPB', 'HPE': 'HPB',
            'HCB': 'HCB', 'HCE': 'HCB',
        }
        current_powers = {}
        
        for i in range(1, 9):
            classe_temporelle_key = f"Classe temporelle {i} (grille turpe)"
            value_str = site_data.get(classe_temporelle_key)

            if isinstance(value_str, str) and ':' in value_str:
                parts = value_str.split(':')
                period_name_from_file = parts[0].strip().upper()
                power_part = parts[1].strip()

                if period_name_from_file in period_mapping:
                    internal_period_name = period_mapping[period_name_from_file]
                    try:
                        power_val_match = re.search(r'(\d+[.,]?\d*)', power_part)
                        if power_val_match:
                            puissance_val = float(power_val_match.group(1).replace(',', '.'))
                            current_powers[internal_period_name] = puissance_val
                    except (ValueError, IndexError):
                        pass
        
        return current_powers

    def map_formula_to_internal(self, raw_formula: str) -> str:
        """
        Mappe la formule tarifaire brute vers une clé interne utilisée dans les tarifs.
        """
        raw_upper = str(raw_formula).upper()
        
        # Mapping amélioré pour BT_SUP36
        if "BTSUPCU" in raw_upper or ("BT" in raw_upper and "CU" in raw_upper):
            mapped = "CU"
        elif "BTSUPLU" in raw_upper or ("BT" in raw_upper and "LU" in raw_upper):
            mapped = "LU"
        elif "CU" in raw_upper:
            mapped = "CU"  
        elif "LU" in raw_upper:
            mapped = "LU"
        else:
            # Fallback par défaut
            mapped = "CU"
            st.warning(f"⚠️ Formule non reconnue: {raw_formula}, utilisation du fallback: {mapped}")
        
        return mapped

    def calculate_power_overruns(self, periods: Dict, puissances: Dict, tension_level: str, formula: str, time_step_hours: float) -> float:
        """Calcule les pénalités de dépassement de puissance."""
        cmdps_total = 0
        if not puissances:
            return 0

        for period, data in periods.items():
            if period in puissances:
                p_souscrite = puissances.get(period, 0)
                if p_souscrite == 0: 
                    continue

                p_max_period = data['Valeur'].max() / 1000  # Conversion en kW

                if p_max_period > p_souscrite:
                    if tension_level == 'HTA':
                        if not isinstance(data.index, pd.DatetimeIndex):
                            data_indexed = data.set_index('datetime')
                        else:
                            data_indexed = data

                        power_10min_avg = data_indexed['Valeur'].resample('10T').mean() / 1000
                        exceedances = (power_10min_avg - p_souscrite).clip(lower=0)
                        sum_of_squared_exceedances = (exceedances**2).sum()

                        if sum_of_squared_exceedances > 0:
                            tarif_puissance = self.tariffs["HTA_PUISSANCE"].get(formula, {}).get(period, 0)
                            cmdps_period = self.tariffs["CMDPS_HTA_COEFF"] * tarif_puissance * np.sqrt(sum_of_squared_exceedances)
                            cmdps_total += cmdps_period
                    else: # BT_SUP36
                        nb_depassements_raw = len(data[data['Valeur'] > p_souscrite * 1000])
                        heures_depassement = nb_depassements_raw * time_step_hours
                        cmdps_period = self.tariffs["CMDPS_BT_SUP36"] * heures_depassement
                        cmdps_total += cmdps_period
        
        return cmdps_total

    def calculate_energy_cost(self, periods: Dict, site_data: pd.Series, formula: str, time_step_hours: float) -> float:
        """Calcule la composante énergie variable du TURPE."""
        tension_level = detect_tension_level(site_data)
        cs_energie = 0

        if tension_level == 'HTA':
            tarif_e = self.tariffs["HTA_ENERGIE"].get(formula, {})
            for period, data in periods.items():
                if period in tarif_e:
                    energie_kwh = (data['Valeur'].sum() * time_step_hours) / 1000
                    cost_period = tarif_e[period] * energie_kwh / 100
                    cs_energie += cost_period
        elif tension_level == 'BT_SUP36':
            tarif_e = self.tariffs["BT_SUP36_ENERGIE"].get(formula, {})
            if not tarif_e:
                st.error(f"❌ Aucun tarif d'énergie trouvé pour BT_SUP36/{formula}")
            for period, data in periods.items():
                if period in tarif_e:
                    energie_kwh = (data['Valeur'].sum() * time_step_hours) / 1000
                    cost_period = tarif_e[period] * energie_kwh / 100
                    cs_energie += cost_period

        return cs_energie

    def calculate_cost(self, site_data: pd.Series, periods: Dict, formula: str, puissances: Dict[str, float], time_step_hours: float, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """Calcule le coût total TURPE pour une configuration donnée, en proratisant les coûts fixes."""
        tension_level = detect_tension_level(site_data)

        # --- Calcul du prorata temporis ---
        days_in_period = (end_date - start_date).days + 1
        is_leap = start_date.year % 4 == 0 and (start_date.year % 100 != 0 or start_date.year % 400 == 0)
        days_in_year = 366 if is_leap else 365
        prorata_factor = days_in_period / days_in_year

        # --- Composantes fixes annuelles (proratisées) ---
        cg_annual = self.tariffs[f"CG_{tension_level}"].get("UNIQUE", 0)
        cc_annual = self.tariffs[f"CC_{tension_level}"]

        cg = cg_annual * prorata_factor
        cc = cc_annual * prorata_factor

        cs_puissance_annual = 0

        if tension_level == 'HTA':
            tarif_p = self.tariffs["HTA_PUISSANCE"].get(formula, {})
            periods_order = ['POINTE', 'HPH', 'HCH', 'HPB', 'HCB']
            p_prev = 0
            for period in periods_order:
                if period in puissances and period in tarif_p:
                    p_current = puissances.get(period, p_prev)
                    cost_period = tarif_p[period] * (p_current - p_prev)
                    cs_puissance_annual += cost_period
                    p_prev = p_current

        elif tension_level == 'BT_SUP36':
            tarif_p = self.tariffs["BT_SUP36_PUISSANCE"].get(formula, {})
            if not tarif_p:
                st.error(f"❌ Aucun tarif de puissance trouvé pour BT_SUP36/{formula}")
            
            periods_order = ['HPH', 'HCH', 'HPB', 'HCB']
            p_prev = 0
            for period in periods_order:
                if period in puissances and period in tarif_p:
                    p_current = puissances.get(period, p_prev)
                    cost_period = tarif_p[period] * (p_current - p_prev)
                    cs_puissance_annual += cost_period
                    p_prev = p_current

        cs_energie = self.calculate_energy_cost(periods, site_data, formula, time_step_hours)

        # Proratiser la composante de soutirage puissance
        cs_puissance = cs_puissance_annual * prorata_factor

        # Calcul des pénalités (ne sont pas proratisées car basées sur des événements réels)
        cmdps = self.calculate_power_overruns(periods, puissances, tension_level, formula, time_step_hours)
        cer = 0  # Énergie réactive non calculée pour l'instant

        total = cg + cc + cs_puissance + cs_energie + cmdps + cer
        
        result = {
            'composante_gestion': cg,
            'composante_comptage': cc,
            'composante_soutirage_puissance': cs_puissance,
            'composante_soutirage_energie': cs_energie,
            'penalites_depassement': cmdps,
            'composante_energie_reactive': cer,
            'total': total
        }
        
        return result

    def calculate_current_cost(self, site_data: pd.Series, consumption_data: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Optional[Dict]:
        """Calcule le coût TURPE avec la configuration actuelle du site pour une période donnée."""
        raw_formula = site_data.get("Formule tarifaire acheminement")
        
        if not raw_formula or pd.isna(raw_formula):
            return None

        current_formula = self.map_formula_to_internal(raw_formula)
        current_powers = self.get_current_powers(site_data)
        if not current_powers:
            return None

        periods, _, time_step_hours = get_time_periods(consumption_data, site_data)
        if not periods:
            return None

        # Passer les dates pour le calcul proratisé
        current_cost_details = self.calculate_cost(site_data, periods, current_formula, current_powers, time_step_hours, start_date, end_date)
        
        return current_cost_details
