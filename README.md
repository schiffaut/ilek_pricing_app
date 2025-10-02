# Application de Pricing d'Offre d'Électricité BtB

**Version** : 1.0.0 | **Dernière mise à jour** : Octobre 2025

---

# Partie 1 : Documentation Fonctionnelle

## 📋 Description

Cette application Streamlit permet de réaliser le pricing d'offres d'électricité pour le segment Business-to-Business (BtB), segment C2-C4, pricing sur mesure. Elle est conçue pour aider les utilisateurs à réaliser les cotations sur la base d'un périmètre, d'un historique de consommation, des prix de l'énergie et des différents coûts et taxes applicables.

## 🚀 Fonctionnalités

- **Paramètres de Cotation** : Définition des détails du client, de la période de fourniture et du type d'offre (prix fixe, par PHS, etc.).
- **Gestion des Données Sites** : Import facile des fichiers de périmètre et des historiques de consommation sur la base du format ENEDIS (C68 pour le périmètre et R63 pour l'historique de consommation).
- **Modification des sites** : Possibilité de mettre à jour une partie du périmètre sur les informations suivantes : date de démarrage, date de fin de fourniture et code NAF pour permettre un pricing spécifique sur les sites se distinguant du reste du périmètre.
- **Courbe de Prix** : Import des courbes de prix forward, sur un pas de temps horaire ou sur un pas de temps 15'.
- **Prévision de Consommation** : Utilisation de modèles d'intelligence artificielle (XGBoost) pour prévoir la consommation future, au pas de temps 15' ou horaire.
- **Paramètres tarifaires** : Possibilité de définir les coûts spécifiques applicables pour la cotation : marge commerciale, coût par année des GO, GC, CEE, primes de risques. Il est possible de charger des primes de risques spécifiques aux différents sites par import de fichier. Enfin, un calcul automatique est développé pour la prime de validité de l'offre, calcul écrasant la valeur par défaut lorsqu'il est lancé
- **Calcul de Pricing** : Calcul détaillé des coûts par site, incluant les composantes réglementaires et les taxes.
- **Export des Résultats** : Export des résultats aux formats Excel d'un côté, et JSON + CSV de l'autre côté (incluant les prévisions de consommation, la courbe de prix utilisée et les différents coûts unitaires et calculs par période).
- **Référentiel** : Définition des différents coûts à utiliser par défaut dans l'outil de cotation, modifiable par l'utilisateur (valeurs enregistrées dans l'application pour être réutilisées ultérieurement).

## 📖 Guide d'Utilisation

Pour tirer le meilleur parti de l'application, suivez ces étapes :
1.  **Paramètres de Cotation** : Commencez par entrer les informations du client et définir la période de l'offre.
2.  **Données Sites** : Chargez les fichiers de données de vos sites et de leur consommation, et modifiez le cas échéant les informations pour chaque site.
3.  **Courbe de Prix** : Chargez la courbe des prix du marché pour la période souhaitée.
4.  **Prévision** : Lancez la prévision de consommation pour obtenir des informations sur l'utilisation future.
5.  **Tarifs** : Définissez les tarifs spécifiques à appliquer pour la cotation.
6.  **Résultats** : Analysez les résultats détaillés dans l'interface de l'application.
7.  **Export** : Exportez les données pour une analyse plus poussée ou pour les partager.

## ⚠️ Contrôles

Les contrôles suivants sont effectués :
1.  La date de fin de validité doit être postérieure à la date de cotation.
2.  Veuillez renseigner une raison sociale pour générer un ID de cotation.
3.  Correspondance entre les PRMs du CSV des consommations et le CSV du périmètre.
4.  Historique de consommation insuffisant. Moins de 9 mois de données disponibles : la prévision pour ce site sera ignorée.
5.  Attention : Historique de consommation limité, entre 9 et 12 mois. La précision de la prévision pourrait être affectée par le manque de données saisonnières complètes. Recommandation : au moins 12 mois d'historique.
6.  Incohérence de granularité entre la courbe de prix et la prévision.
7.  La période de fourniture du site n'est pas incluse dans la période de cotation globale.
8.  La courbe de prix ne couvre pas toute la période de fourniture.
9.  La prévision n'a pas été générée pour la même que la durée de fourniture. Veuillez relancer la prévision.

## 📞 Support

En cas de problème :
1.  Vérifiez cette documentation.
2.  Consultez les messages d'erreur dans l'interface.
3.  Utilisez les données d'exemple fournies pour tester les fonctionnalités.

---

# Partie 2 : Documentation Technique

## 📁 Structure du Projet

```
pricing-app/
│
├── main.py                    # Application principale Streamlit
├── assets/                    # Logos Fournisseur et Copilote Energie
├── data/                      # Fichier Excel nommé "naf_codes.xlsx" reprenant les différents codes NAF et l'éligibilité ou non aux CEE
├── modules/
│   ├── data_loader.py         # Chargement et validation des données
├── src/
│   ├── helpers.py             # Récupération des informations relatives au TURPE (plages, FTA, etc.), normalisation des caractères spéciaux
│   ├── risk_premiums.py       # Calcul des primes de risques
│   ├── turpe_calculator.py    # Calcul du TURPE
├── requirements.txt           # Dépendances Python
├── tariffs_config.json        # Configuration de l'ensemble des tarifs, modifiable via le référentiel de l'interface
└── README.md                  # Documentation
```

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd pricing-app
   ```

2. **Créer un environnement virtuel** (recommandé)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Lancer l'application**
   ```bash
   streamlit run main.py
   ```

   L'application sera accessible à l'adresse : `http://localhost:8501`

## 📊 Format des Données d'Entrée

### Fichier Périmètre
Fichier C68 d'ENEDIS au format CSV

### Fichier Consommation
Fichier R63 d'ENEDIS au format CSV

### Fichier Courbe de Prix
Format CSV avec séparateur point-virgule (`;`) :
```csv
datetime;prix_euro_mwh
2024-01-01 00:00:00;45.25
```

### Fichier Primes de risque par site
Format CSV avec séparateur point-virgule (`;`) :
```csv
PRM;Délai de validité;Coût d'équilibrage;Risque volume;Délai de Paiement
50026862835537;1.5;2.5;3.0;0.35
50087210157666;1.2;2.0;;0.30
50090249231433;;;2.5;
```

## 🤖 Modèle de Prévision

- **Algorithme** : XGBoost
- **Features Utilisées** : Temporelles (heure, jour, mois), cycliques (sin/cos), lag (J-1, J-7).
- **Validation** : Train/test set configurable.
- **Métriques** : MAE (Mean Absolute Error), R² Score.

## 🧮 Calculs de Pricing

- **Coût de l'Électron** : `Σ(Consommation_h × Prix_h) / Σ(Consommation_h)`
- **Composantes Réglementaires** : CEE, Accise, TVA.
- **Formule Finale** : `Prix TTC = (Coût Électron + CEE + Accise) × (1 + TVA)`

## 🐛 Dépannage

- **Erreur d'encodage CSV** : Assurez-vous que le fichier est encodé en UTF-8.
- **Données météorologiques indisponibles** : Vérifiez votre connexion internet.
- **Erreur de format de date** : Utilisez le format YYYY-MM-DD HH:MM:SS.

## 📄 Licence

Ce projet est développé par COPILOTE Energie avec conservation de la propriété intellectuelle sur les fonctionnalités standards d'un pricer BtB pour des offres standards (import et gestion d'un périmètre, prévision de consommation, modalités de calcul du prix de l'énergie, de l'acheminement et des taxes).
