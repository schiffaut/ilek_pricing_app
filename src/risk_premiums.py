import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from scipy import stats
from typing import Dict, Optional, Tuple
import json


class RiskPremiumCalculator:
    """Calculateur des primes de risque pour les offres d'électricité"""
    
    def __init__(self):
        self.premium_types = [
            "Délai de validité",
            "Coût d'équilibrage",
            "Coût du profil",
            "Risque volume",
            "Fluctuation du parc",
            "Recalage Prix Live",
            "Délai de Paiement",
            "Risque Crédit"
        ]
    
    def calculate_validity_period_premium(
        self, 
        validity_hours: int,
        market_price: float,
        volatility_pct: float,
        risk_coverage_pct: float,
        price_curve: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Calcule la prime de risque pour le délai de validité
        
        Args:
            validity_hours: Durée de validité en heures
            market_price: Prix de marché de référence (€/MWh)
            volatility_pct: Volatilité annuelle en %
            risk_coverage_pct: Niveau de risque à couvrir en %
            price_curve: Courbe de prix historique pour calibrage
            
        Returns:
            Dict avec les détails du calcul
        """
        
        # Conversion de la volatilité annuelle en volatilité par heure
        annual_volatility = volatility_pct / 100
        hourly_volatility = annual_volatility / np.sqrt(8760)  # 8760 heures par an
        
        # Volatilité pour la période de validité
        period_volatility = hourly_volatility * np.sqrt(validity_hours)
        
        # Calcul du quantile de risque
        confidence_level = risk_coverage_pct / 100
        z_score = stats.norm.ppf(0.5 + confidence_level/2)
        
        # Prime de risque basée sur le VaR (Value at Risk)
        price_var = market_price * period_volatility * z_score
        
        # Ajustement pour la liquidité (plus la période est longue, plus le risque augmente)
        liquidity_factor = 1 + (validity_hours / (24 * 30)) * 0.1  # +10% par mois
        
        premium_eur_mwh = price_var * liquidity_factor
        
        # Calibrage avec données historiques si disponibles
        if price_curve is not None and len(price_curve) > validity_hours:
            historical_volatility = self._calculate_historical_volatility(price_curve, validity_hours)
            adjustment_factor = historical_volatility / (market_price * period_volatility)
            premium_eur_mwh *= adjustment_factor
        
        return {
            'premium_eur_mwh': premium_eur_mwh,
            'calculation_details': {
                'validity_hours': validity_hours,
                'market_price': market_price,
                'annual_volatility_pct': volatility_pct,
                'period_volatility_pct': period_volatility * 100,
                'confidence_level_pct': risk_coverage_pct,
                'z_score': z_score,
                'price_var': price_var,
                'liquidity_factor': liquidity_factor,
                'final_premium': premium_eur_mwh
            }
        }
    
    def _calculate_historical_volatility(self, price_curve: pd.DataFrame, validity_hours: float) -> float:
        """Calcule la volatilité historique sur des périodes glissantes - VERSION CORRIGÉE"""
        
        # BUG FIX: Convertir validity_hours en entier pour l'indexation et range()
        validity_hours_int = int(validity_hours)

        if len(price_curve) < validity_hours_int * 2:
            return 0
        
        rolling_changes = []
        for i in range(len(price_curve) - validity_hours_int):
            start_price = price_curve.iloc[i]['prix_euro_mwh']
            end_price = price_curve.iloc[i + validity_hours_int]['prix_euro_mwh']
            
            if start_price <= 0:
                continue
                
            pct_change = (end_price - start_price) / start_price
            rolling_changes.append(pct_change)
        
        if len(rolling_changes) == 0:
            return 0
        
        # ✅ CORRECTION: retourner SEULEMENT l'écart-type (pas × prix moyen)
        return np.std(rolling_changes)
    
    def load_premiums_from_file(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Charge les primes de risque spécifiques par PRM depuis un fichier CSV
        
        Format attendu: PRM, Délai_de_validité, Coût_équilibrage, etc.
        """
        
        try:
            # Lecture du fichier
            try:
                df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            
            # Standardisation des colonnes
            df.columns = df.columns.str.strip()
            
            # Vérification de la colonne PRM
            prm_columns = [col for col in df.columns if 'prm' in col.lower() or 'point' in col.lower()]
            if not prm_columns:
                st.error("Colonne PRM non trouvée dans le fichier")
                return None
            
            # Renommage de la colonne PRM
            df = df.rename(columns={prm_columns[0]: 'PRM'})
            
            # Nettoyage des données
            df['PRM'] = df['PRM'].astype(str).str.strip()
            df = df.dropna(subset=['PRM'])
            
            # Conversion des colonnes de primes en numérique
            premium_columns = []
            for col in df.columns:
                if col != 'PRM':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        premium_columns.append(col)
                    except:
                        pass
            
            st.success(f"Fichier de primes chargé : {len(df)} PRMs avec {len(premium_columns)} types de primes")
            
            return df
            
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier de primes : {str(e)}")
            return None
    
    def get_site_premiums(self, prm: str, premiums_df: Optional[pd.DataFrame], default_premiums: Dict) -> Dict:
        """
        Récupère les primes de risque pour un site donné
        
        Args:
            prm: Identifiant du site
            premiums_df: DataFrame avec les primes spécifiques par PRM
            default_premiums: Primes par défaut
            
        Returns:
            Dict avec les primes à appliquer
        """
        
        site_premiums = default_premiums.copy()
        
        # Si des primes spécifiques sont définies pour ce PRM
        if premiums_df is not None and prm in premiums_df['PRM'].values:
            site_row = premiums_df[premiums_df['PRM'] == prm].iloc[0]
            
            # Mise à jour avec les valeurs spécifiques
            for premium_type in self.premium_types:
                # Recherche de la colonne correspondante (avec gestion des variations de noms)
                matching_cols = [col for col in premiums_df.columns 
                               if premium_type.lower().replace(' ', '_') in col.lower().replace(' ', '_')]
                
                if matching_cols:
                    col_name = matching_cols[0]
                    if pd.notna(site_row[col_name]):
                        site_premiums[premium_type] = float(site_row[col_name])
        
        return site_premiums
    
    def calculate_total_premium(self, premiums: Dict) -> float:
        """Calcule la prime totale en €/MWh"""
        return sum(premiums.values())
    
    def get_premium_breakdown_df(self, premiums: Dict) -> pd.DataFrame:
        """Retourne un DataFrame avec le détail des primes"""
        
        breakdown_data = []
        for premium_type, value in premiums.items():
            breakdown_data.append({
                'Type de Prime': premium_type,
                'Valeur (€/MWh)': value
            })
        
        # Ajout du total
        breakdown_data.append({
            'Type de Prime': 'TOTAL PRIMES',
            'Valeur (€/MWh)': self.calculate_total_premium(premiums)
        })
        
        return pd.DataFrame(breakdown_data)


class CTACalculator:
    """Calculateur de la Contribution Tarifaire d'Acheminement (CTA)"""
    
    def __init__(self, cta_rate: float = 0.2193):
        """
        Args:
            cta_rate: Taux de CTA (21,93% par défaut)
        """
        self.cta_rate = cta_rate
    
    def calculate_cta(self, turpe_details: Dict) -> Dict:
        """
        Calcule la CTA sur les composantes fixes du TURPE
        
        Args:
            turpe_details: Détails du calcul TURPE
            
        Returns:
            Dict avec le détail du calcul CTA
        """
        
        if not turpe_details:
            return {
                'composante_gestion_cta': 0,
                'composante_comptage_cta': 0,
                'composante_soutirage_puissance_cta': 0,
                'total_cta': 0,
                'taux_cta': self.cta_rate
            }
        
        # Calcul de la CTA sur les composantes fixes
        cg_cta = turpe_details.get('composante_gestion', 0) * self.cta_rate
        cc_cta = turpe_details.get('composante_comptage', 0) * self.cta_rate
        cs_puissance_cta = turpe_details.get('composante_soutirage_puissance', 0) * self.cta_rate
        
        # La composante soutirage énergie et les pénalités ne sont pas soumises à CTA
        total_cta = cg_cta + cc_cta + cs_puissance_cta
        
        return {
            'composante_gestion_cta': cg_cta,
            'composante_comptage_cta': cc_cta,
            'composante_soutirage_puissance_cta': cs_puissance_cta,
            'total_cta': total_cta,
            'taux_cta': self.cta_rate,
            'base_calcul': {
                'composante_gestion': turpe_details.get('composante_gestion', 0),
                'composante_comptage': turpe_details.get('composante_comptage', 0),
                'composante_soutirage_puissance': turpe_details.get('composante_soutirage_puissance', 0)
            }
        }


# Les fonctions d'interface render_risk_premium_interface et render_cta_interface
# ont été déplacées et intégrées directement dans les onglets respectifs de main.py
# pour une meilleure gestion de l'état et de la logique de l'interface.
# Ce fichier ne contient plus que la logique de calcul.