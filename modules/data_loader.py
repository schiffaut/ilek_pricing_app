import pandas as pd
import streamlit as st
from typing import Optional

def load_enedis_perimeter(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge et valide un fichier de périmètre ENEDIS avec gestion robuste des formats
    
    Args:
        uploaded_file: Fichier uploadé via Streamlit
    
    Returns:
        DataFrame contenant les données du périmètre ou None si erreur
    """
    try:
        # Tentative de lecture avec différents encodages et séparateurs
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
        
        # Standardisation des noms de colonnes
        df.columns = df.columns.str.lower().str.strip()
        
        # Mapping étendu des colonnes courantes ENEDIS
        column_mapping = {
            'point_reference_mesure': 'prm',
            'identifiant prm': 'prm',  # Format avec espace minuscule
            'identifiant_prm': 'prm',  # Format avec underscore
            'prm_id': 'prm',
            'lat': 'latitude',
            'lng': 'longitude',
            'lon': 'longitude',
            'code_naf_ape': 'code_naf',
            'code naf': 'code_naf',  # Nouveau mapping
            'naf': 'code_naf',
            'raison_sociale': 'company_name',
            'denomination sociale du client final': 'company_name',  # Nouveau mapping
            'nom': 'company_name',
            'adresse': 'address',
            'adresse ligne 1': 'address',  # Nouveau mapping
            'ville': 'city',
            'commune': 'city',  # Nouveau mapping
            'code_postal': 'postal_code',
            'code postal': 'postal_code',  # Nouveau mapping
            'puissance_souscrite': 'subscribed_power',
            'puissance souscrite': 'subscribed_power',  # Nouveau mapping
            'tarif': 'tariff',
            'code tarif acheminement': 'tariff',  # Nouveau mapping
            'numero siret': 'siret',
            'numero siren': 'siren',
            'secteur activite': 'activity_sector'
        }
        
        # Application du mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
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
        
        # Conversion de la puissance souscrite si présente
        if 'subscribed_power' in df.columns:
            # Gestion des formats avec unités (kW, MW, etc.)
            df['subscribed_power'] = df['subscribed_power'].astype(str).str.replace(' kW', '').str.replace(' MW', '').str.replace(',', '.')
            df['subscribed_power'] = pd.to_numeric(df['subscribed_power'], errors='coerce')
        
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
        
        st.success(f"Fichier périmètre chargé avec succès : {len(df)} sites")
        
        # Affichage d'informations sur les données
        if 'latitude' in df.columns and 'longitude' in df.columns:
            valid_coords = df.dropna(subset=['latitude', 'longitude'])
            st.info(f"Sites avec coordonnées GPS valides : {len(valid_coords)}/{len(df)}")
        
        if 'subscribed_power' in df.columns:
            valid_power = df.dropna(subset=['subscribed_power'])
            if len(valid_power) > 0:
                st.info(f"Puissance souscrite moyenne : {valid_power['subscribed_power'].mean():.1f} kW")
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier périmètre : {str(e)}")
        return None


def parse_time_step(time_step_str: str) -> int:
    """
    Convertit un pas de temps au format ISO 8601 en minutes
    
    Args:
        time_step_str: Pas de temps (ex: "PT5M", "PT10M", "PT30M", "PT1H")
    
    Returns:
        Nombre de minutes (défaut: 60 si non reconnu)
    """
    if pd.isna(time_step_str) or time_step_str is None:
        return 60  # Défaut horaire
        
    time_step_str = str(time_step_str).upper().strip()
    
    time_step_mapping = {
        'PT5M': 5,
        'PT10M': 10,
        'PT15M': 15,
        'PT30M': 30,
        'PT1H': 60,
        'PT60M': 60,
        '5M': 5,
        '10M': 10,
        '15M': 15,
        '30M': 30,
        '1H': 60,
        '60M': 60,
        'NULL': 60,  # Valeur null dans les données
        'NAN': 60
    }
    
    return time_step_mapping.get(time_step_str, 60)  # Par défaut horaire


def load_enedis_consumption(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Charge et valide un fichier de consommation ENEDIS avec gestion robuste des formats
    et des différentes granularités temporelles
    
    Args:
        uploaded_file: Fichier uploadé via Streamlit
    
    Returns:
        DataFrame contenant les données de consommation ou None si erreur
    """
    try:
        # Tentative de lecture avec différents encodages et séparateurs
        try:
            df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=';', encoding='latin-1')
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file, sep=',', encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=',', encoding='latin-1')
        
        # Standardisation des noms de colonnes
        df.columns = df.columns.str.lower().str.strip()
        
        # Mapping étendu des colonnes courantes ENEDIS
        column_mapping = {
            'point_reference_mesure': 'prm',
            'identifiant prm': 'prm',  # Nouveau mapping pour format ENEDIS standard
            'prm_id': 'prm',
            'identifiant_prm': 'prm',
            'horodate': 'datetime',
            'date_heure': 'datetime',
            'timestamp': 'datetime',
            'date': 'datetime',
            'valeur': 'consumption_raw',  # Changé pour traiter les unités
            'consommation': 'consumption_raw',
            'consumption': 'consumption_raw',
            'energy': 'consumption_raw',
            'kwh': 'consumption_raw',
            'pas': 'time_step',
            'step': 'time_step',
            'intervalle': 'time_step',
            'unité': 'unit',
            'unite': 'unit',
            'grandeur physique': 'physical_quantity',
            'grandeur métier': 'business_quantity',
            'grandeur metier': 'business_quantity',
            'etape métier': 'business_step',
            'etape metier': 'business_step',
            'nature': 'nature'
        }
        
        # Application du mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Validation des colonnes obligatoires
        required_columns = ['prm', 'datetime', 'consumption_raw']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Colonnes manquantes dans le fichier consommation : {missing_columns}")
            st.error("Colonnes détectées dans le fichier :")
            st.write(list(df.columns))
            return None
        
        # Nettoyage et conversion des données
        df = df.dropna(subset=['prm', 'datetime', 'consumption_raw'])
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
        
        # Gestion des unités et conversion en kWh
        df['consumption_kWh'] = pd.to_numeric(df['consumption_raw'], errors='coerce')
        
        # Conversion selon l'unité si présente
        if 'unit' in df.columns:
            unit_values = df['unit'].unique()
            st.info(f"Unités détectées : {list(unit_values)}")
            
            # Conversion des watts en kWh selon le pas de temps
            if 'W' in unit_values:
                # Récupération du pas de temps
                if 'time_step' in df.columns:
                    df['time_step_minutes'] = df['time_step'].apply(parse_time_step)
                    # Conversion W -> kWh : (W * minutes) / (1000 * 60)
                    mask_watts = df['unit'] == 'W'
                    df.loc[mask_watts, 'consumption_kWh'] = (
                        df.loc[mask_watts, 'consumption_raw'] * df.loc[mask_watts, 'time_step_minutes']
                    ) / (1000 * 60)
                    
                    st.info("Conversion des valeurs en Watts vers kWh effectuée selon le pas de temps")
                else:
                    st.warning("Unité en Watts détectée mais pas de pas de temps - conversion impossible")
            
            # Conversion kW -> kWh
            elif 'kW' in unit_values:
                if 'time_step' in df.columns:
                    df['time_step_minutes'] = df['time_step'].apply(parse_time_step)
                    mask_kw = df['unit'] == 'kW'
                    df.loc[mask_kw, 'consumption_kWh'] = (
                        df.loc[mask_kw, 'consumption_raw'] * df.loc[mask_kw, 'time_step_minutes'] / 60
                    )
                    st.info("Conversion des valeurs en kW vers kWh effectuée selon le pas de temps")
        
        # Nettoyage final
        df = df.dropna(subset=['consumption_kWh'])
        
        # Validation des valeurs de consommation
        negative_values = (df['consumption_kWh'] < 0).sum()
        if negative_values > 0:
            st.warning(f"Attention : {negative_values} valeurs de consommation négatives détectées")
        
        # Tri par datetime
        df = df.sort_values(['prm', 'datetime'])
        
        # Détection du pas de temps effectif dans les données
        if len(df) > 1:
            time_diff = df.groupby('prm')['datetime'].diff().dropna()
            if not time_diff.empty:
                mode_diff = time_diff.mode()
                if not mode_diff.empty:
                    time_step = mode_diff[0]
                    if time_step == pd.Timedelta(hours=1):
                        detected_step = "Horaire"
                    elif time_step == pd.Timedelta(minutes=30):
                        detected_step = "30 minutes"
                    elif time_step == pd.Timedelta(minutes=15):
                        detected_step = "15 minutes"
                    elif time_step == pd.Timedelta(minutes=10):
                        detected_step = "10 minutes"
                    elif time_step == pd.Timedelta(minutes=5):
                        detected_step = "5 minutes"
                    else:
                        detected_step = f"{time_step}"
                    
                    st.info(f"Pas de temps effectif détecté : {detected_step}")
        
        # Affichage des informations sur le pas de temps déclaré
        if 'time_step' in df.columns:
            declared_steps = df['time_step'].value_counts()
            st.info(f"Pas de temps déclarés : {dict(declared_steps)}")
        
        st.success(f"Fichier consommation chargé avec succès : {len(df)} mesures")
        
        # Statistiques sur les données
        date_min = df['datetime'].min()
        date_max = df['datetime'].max()
        consumption_mean = df['consumption_kWh'].mean()
        consumption_max = df['consumption_kWh'].max()
        
        st.info(f"Période : {date_min.strftime('%Y-%m-%d')} à {date_max.strftime('%Y-%m-%d')}")
        st.info(f"Consommation moyenne : {consumption_mean:.2f} kWh, Maximum : {consumption_max:.2f} kWh")
        
        # Détection des différentes granularités par PRM
        prm_stats = df.groupby('prm').agg({
            'datetime': ['min', 'max', 'count'],
            'consumption_kWh': ['mean', 'sum']
        }).round(2)
        
        if len(df['prm'].unique()) <= 10:  # Affichage seulement si peu de PRMs
            st.info("Statistiques par PRM :")
            st.dataframe(prm_stats)
        
        return df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier consommation : {str(e)}")
        return None


def validate_data_consistency(perimeter_df: pd.DataFrame, consumption_data: dict) -> dict:
    """
    Valide la cohérence entre les données de périmètre et de consommation
    
    Args:
        perimeter_df: DataFrame du périmètre
        consumption_data: Dictionnaire des DataFrames de consommation par PRM
    
    Returns:
        Dictionnaire contenant les résultats de validation
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
        
        # Validation des périodes de données pour chaque PRM
        data_quality_issues = []
        for prm, df_consumption in consumption_data.items():
            if prm in common_prms and len(df_consumption) > 0:
                date_min = df_consumption['datetime'].min()
                date_max = df_consumption['datetime'].max()
                data_points = len(df_consumption)
                
                # Calcul du nombre de points attendus selon la granularité
                time_diff = df_consumption['datetime'].diff().dropna()
                if not time_diff.empty:
                    mode_diff = time_diff.mode()
                    if not mode_diff.empty:
                        step_minutes = mode_diff[0].total_seconds() / 60
                        expected_points = int((date_max - date_min).total_seconds() / (step_minutes * 60)) + 1
                        
                        completeness_ratio = data_points / expected_points if expected_points > 0 else 0
                        
                        if completeness_ratio < 0.8:  # Moins de 80% de données
                            data_quality_issues.append(
                                f"PRM {prm} : complétude {completeness_ratio:.1%} ({data_points}/{expected_points} points)"
                            )
        
        if data_quality_issues:
            validation_results['warnings'].extend(data_quality_issues[:5])  # Limiter l'affichage
            if len(data_quality_issues) > 5:
                validation_results['warnings'].append(f"... et {len(data_quality_issues) - 5} autres problèmes de qualité")
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Erreur lors de la validation : {str(e)}")
    
    return validation_results