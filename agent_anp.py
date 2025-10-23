import pandas as pd
import ollama
import sys
import re
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from difflib import get_close_matches
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HybridPortAgent:
    def attente_moyenne_rade_accostage(self) -> float:
        """Calcule le temps d'attente moyen entre l'arrivée en rade et l'accostage au quai, en filtrant les valeurs aberrantes (0-240h)."""
        if 'DATETIME_RADE' in self.df.columns and 'DATETIME_ACCOSTAGE' in self.df.columns:
            df_valid = self.df.dropna(subset=['DATETIME_RADE', 'DATETIME_ACCOSTAGE']).copy()
            df_valid['DATETIME_RADE'] = pd.to_datetime(df_valid['DATETIME_RADE'], errors='coerce')
            df_valid['DATETIME_ACCOSTAGE'] = pd.to_datetime(df_valid['DATETIME_ACCOSTAGE'], errors='coerce')
            attente = (df_valid['DATETIME_ACCOSTAGE'] - df_valid['DATETIME_RADE']).dt.total_seconds() / 3600
            attente_filtrée = attente[(attente >= 0) & (attente <= 240)]
            return attente_filtrée.mean()
        else:
            raise ValueError("Colonnes nécessaires manquantes dans le DataFrame.")
    """Agent portuaire hybride avec traitement correct des dates complexes"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df_original = None
        self.df = None
        self.error_history = []
        self.performance_stats = {"total_queries": 0, "successful_queries": 0, "avg_response_time": 0}
        
        # Column mapping from original system
        self.column_mapping = {
            'armateur': 'LIBELLE_ARMATEUR',
            'navire': 'NOM_NAVIRE',
            'type navire': 'TYPE_NAVIRE',
            'pavillon': 'PAVILLON',
            'destination': 'DESTINATION',
            'provenance': 'PROVENANCE',
            'longueur': 'LONGEUR_NAVIRE',
            'tonnage import': 'TONNAGE_IMPORT',
            'tonnage export': 'TONNAGE_EXPORT',
            'année': 'ANNEE',
            'sejour quai': 'SEJOUR_QUAI_HEURE',
            'sejour port': 'SEJOUR_PORT_HEURE',
            'temps attente': 'TEMPS_ATTENTE_ACCOSTAGE_HEURES',
            'temps accostage': 'TEMPS_ENTREE_ACCOSTAGE_HEURES'
        }
        
        # Enhanced analysis templates
        self.analysis_templates = self._load_enhanced_templates()
        self.column_descriptions = self._load_column_descriptions()
        
        # Initialize
        self.load_and_prepare_data()
        
    def _load_column_descriptions(self) -> str:
        """Load detailed column descriptions"""
        return """
Voici la signification des colonnes du DataFrame :
- ID : Identifiant unique de l'enregistrement
- CODE_SOCIETE : Code de la société opératrice ou armateur
- NUMERO_ESCALE : Numéro unique de l'escale
- NUMERO_LLOYD : Numéro Lloyd's du navire (identifiant international)
- NOM_NAVIRE : Nom du navire
- TYPE_NAVIRE : Type de navire (pêche, porte-conteneur, etc.)
- PAVILLON : Pays du pavillon du navire
- LONGEUR_NAVIRE : Longueur du navire (mètres)
- TIRANT_EAU_NAVIRE : Tirant d'eau du navire (mètres)
- JAUGE_BRUTE : Jauge brute du navire
- CONSIGNATAIRE : Société consignataire
- OPERATEUR : Opérateur(s) portuaire(s)
- PROVENANCE : Port de provenance
- PAYS_PROVENANCE : Pays de provenance
- DESTINATION : Port de destination
- PAYS_DESTINATION : Pays de destination
- LIBELLE_ARMATEUR : Nom de l'armateur
- DATE_RADE, HEURE_RADE : Date et heure d'arrivée en rade
- DATE_ENTREE_PORT, HEURE_ENTREE_PORT : Date et heure d'entrée au port
- DATE_ACCOSTAGE, HEURE_ACCOSTAGE : Date et heure d'accostage
- DATE_APP_QUAI, HEURE_APP_QUAI : Date et heure d'appareillage du quai
- DATE_APP_PORT, HEURE_APP_PORT : Date et heure d'appareillage du port
- DATE_ENTREE_MOUILLAGE, HEURE_ENTREE_MOUILLAGE : Date et heure d'entrée au mouillage
- DATE_SORTIE_MOUILLAGE, HEURE_SORTIE_MOUILLAGE : Date et heure de sortie du mouillage
- SEJOUR_QUAI_HEURE : Durée de séjour au quai (heures)
- SEJOUR_FACTURE_HEURE : Durée de séjour facturée (heures)
- SEJOUR_RADE_HEURE : Durée de séjour en rade (heures)
- SEJOUR_PORT_HEURE : Durée de séjour au port (heures)
- SEJOUR_MOUILLAGE_HEURE : Durée de séjour au mouillage (heures)
- LIBELLE_ARMATEUR_DISPOSANT : Armateur disposant du navire
- POSTE_ACCOSTAGE : Poste/quai d'accostage
- TIRANT_EAU_AVANT : Tirant d'eau avant à l'arrivée
- TIRANT_EAU_ARRIERE : Tirant d'eau arrière à l'arrivée
- TIRANT_EAU_SORTIE_AVANT : Tirant d'eau avant au départ
- TIRANT_EAU_SORTIE_ARRIERE : Tirant d'eau arrière au départ
- ANNEE : Année de l'escale
- CAUSE_ATTENTE : Cause d'attente éventuelle
- DUREE_ATTENTE : Durée d'attente (minutes ou heures)
- DATE_MOUVEMENT : Date du mouvement principal
- TYPE_REDUCTION : Type de réduction appliquée
- TYPE_NAVIRE_CODE : Code du type de navire
- PAVILLON_NAVIRE_CODE : Code du pavillon du navire
- Libelle_Ligne : Libellé de la ligne maritime
- MARCHANDISES_ESCALE_DAP_IMPORT : Type de marchandises importées à l'escale
- TONNAGE_MARCHANDISES_ESCALE_DAP_IMPORT : Tonnage importé à l'escale
- MARCHANDISES_ESCALE_DAP_EXPORT : Type de marchandises exportées à l'escale
- TONNAGE_MARCHANDISES_ESCALE_DAP_EXPORT : Tonnage exporté à l'escale
- MARCHANDISES_FACTUREE : Marchandises facturées
- TONNAGE_MARCHANDISES_FACTUREE : Tonnage facturé
- Tonnage_Manifeste_Import : Tonnage manifeste à l'import
- Tonnage_Manifeste_Export : Tonnage manifeste à l'export
- Date_Systeme : Date système d'enregistrement
- NBR_REMORQUEUR_ACCOSTAGE : Nombre de remorqueurs à l'accostage
- NBR_REMORQUEUR_APPAREILLAGE : Nombre de remorqueurs à l'appareillage
- NBR_REMORQUEUR_CHANGEMENT : Nombre de remorqueurs pour changement
- NBR_REMORQUEUR_ASSISTANCE : Nombre de remorqueurs pour assistance
- FACTURE_DM : Facture DM (oui/non)
- FACTURE_DN : Facture DN (oui/non)
- MONTANT_HT_FACTURE_DM : Montant HT de la facture DM
- MONTANT_HT_FACTURE_DN : Montant HT de la facture DN
- DATE_VALIDATION_ESCALE : Date de validation de l'escale
- Existe_Manifeste : Indique si un manifeste existe (oui/non)
- DATE_FACTURE_DM : Date de la facture DM
- DATE_FACTURE_DN : Date de la facture DN
- BOITE_IMPORT : Nombre de boîtes importées (conteneurs)
- TONNAGE_IMPORT : Tonnage importé
- BOITE_EXPORT : Nombre de boîtes exportées (conteneurs)
- TONNAGE_EXPORT : Tonnage exporté
- PRODUCTIVITE_BOITE_HEURE : Productivité en boîtes/heure
- PRODUCTIVITE_TONNAGE_JOUR : Productivité en tonnage/jour
- PRODUCTIVITE_BOITE_HEURE_Port : Productivité en boîtes/heure (port)
- PRODUCTIVITE_TONNAGE_JOUR_PORT : Productivité en tonnage/jour (port)
- PRODUCTIVITE_BOITE_HEURE_Port_NEW : Nouvelle productivité en boîtes/heure (port)
- ESCALE_CABOTAGE : Indique si l'escale est de cabotage (oui/non)
"""
    
    def _load_enhanced_templates(self) -> Dict[str, Dict]:
        templates = {
            'evolution_tonnage_mensuel': {
                'keywords': ['évolution tonnage', 'evolution tonnage', 'tonnage par mois', 'chart évolution tonnage', 'graphique tonnage', 'tonnage mensuel'],
                'validation': lambda df: 'DATETIME_RADE' in df.columns and 'TONNAGE_IMPORT' in df.columns,
                'code_template': """
# Évolution du tonnage importé par mois (graphique)
df_valid = df.dropna(subset=['DATETIME_RADE', 'TONNAGE_IMPORT']).copy()
df_valid['DATETIME_RADE'] = pd.to_datetime(df_valid['DATETIME_RADE'], errors='coerce')
df_valid['Mois'] = df_valid['DATETIME_RADE'].dt.to_period('M').astype(str)
monthly_tonnage = df_valid.groupby('Mois')['TONNAGE_IMPORT'].sum()
result = monthly_tonnage
import matplotlib.pyplot as plt
monthly_tonnage.plot(kind='bar', figsize=(14,6))
plt.title('Évolution du tonnage importé par mois')
plt.ylabel('Tonnage importé')
plt.xlabel('Mois')
plt.tight_layout()
plt.show()
""",
                'visualization': True
            },
            'productivite_terminaux': {
                'keywords': ['productivité', 'productivite', 'terminal', 'poste', 'quai', 'poste_accostage'],
                'validation': lambda df: 'POSTE_ACCOSTAGE' in df.columns and (
                    'PRODUCTIVITE_BOITE_HEURE_Port' in df.columns or 'PRODUCTIVITE_TONNAGE_JOUR_PORT' in df.columns),
                'code_template': """
# Calcul de la productivité moyenne par terminal (poste/quai)
cols = []
if 'PRODUCTIVITE_BOITE_HEURE_Port' in df.columns:
    cols.append('PRODUCTIVITE_BOITE_HEURE_Port')
if 'PRODUCTIVITE_TONNAGE_JOUR_PORT' in df.columns:
    cols.append('PRODUCTIVITE_TONNAGE_JOUR_PORT')
if not cols:
    raise ValueError('Colonnes de productivité non disponibles')

grouped = df.groupby('POSTE_ACCOSTAGE')[cols].mean().fillna(0)
grouped = grouped.sort_values(by=cols, ascending=False)
result = grouped.head(10)

import matplotlib.pyplot as plt
ax = result.plot(kind='bar', figsize=(12,6))
plt.title('Productivité Moyenne par Terminal (Top 10)')
plt.ylabel('Productivité')
plt.xlabel('Terminal (Poste/Quai)')
plt.tight_layout()
plt.show()
""",
                'visualization': True
            },
            'evolution_temporelle': {
                'keywords': ['évolution', 'temporel', 'temps', 'mensuel', 'annuel', 'tendance'],
                'validation': lambda df: 'DATETIME_RADE' in df.columns and df['DATETIME_RADE'].notna().sum() > 0,
                'code_template': """
# Validation des données temporelles
if 'DATETIME_RADE' not in df.columns:
    raise ValueError("Colonne DATETIME_RADE introuvable")

df_valid = df.dropna(subset=['DATETIME_RADE']).copy()
if len(df_valid) == 0:
    raise ValueError("Aucune date valide trouvée")

# Détection de la granularité demandée
question_lower = "{question}".lower()
if any(word in question_lower for word in ['annuel', 'année', 'an']):
    df_valid['PERIODE'] = df_valid['DATETIME_RADE'].dt.year
    title = "Évolution Annuelle des Escales"
elif any(word in question_lower for word in ['mensuel', 'mois']):
    df_valid['PERIODE'] = df_valid['DATETIME_RADE'].dt.to_period('M')
    title = "Évolution Mensuelle des Escales"
else:
    df_valid['PERIODE'] = df_valid['DATETIME_RADE'].dt.to_period('M')
    title = "Évolution Mensuelle des Escales (par défaut)"

result = df_valid.groupby('PERIODE').size().sort_index()
print(f"Période analysée: {result.index.min()} à {result.index.max()}")
print(f"Nombre total d'escales: {result.sum()}")
""",
                'visualization': True
            },
            
            'temps_attente': {
                'keywords': ['temps', 'attente', 'délai', 'durée', 'moyenne'],
                'validation': lambda df: 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in df.columns,
                'code_template': """
# Utilisation de la colonne corrigée pour les temps d'attente
if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in df.columns:
    temps_data = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
    
    # Filtrer les valeurs aberrantes (0 à 240 heures)
    temps_data = temps_data[(temps_data >= 0) & (temps_data <= 240)]
    
    if len(temps_data) == 0:
        raise ValueError("Aucune donnée de temps d'attente valide après filtrage")
    
    result_dict = {
        'moyenne': temps_data.mean(),
        'mediane': temps_data.median(),
        'min': temps_data.min(),
        'max': temps_data.max(),
        'nb_escales': len(temps_data),
        'nb_total': len(df),
        'taux_completude': len(temps_data) / len(df) * 100
    }
    
    print(f"Statistiques de temps d'attente CORRIGÉES (en heures):")
    print(f"- Moyenne: {result_dict['moyenne']:.2f}h")
    print(f"- Médiane: {result_dict['mediane']:.2f}h") 
    print(f"- Minimum: {result_dict['min']:.2f}h")
    print(f"- Maximum: {result_dict['max']:.2f}h")
    print(f"- Nombre d'escales analysées: {result_dict['nb_escales']}")
    print(f"- Taux de complétude: {result_dict['taux_completude']:.1f}%")
    
    result = result_dict['moyenne']
else:
    raise ValueError("Colonne TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE non disponible")
""",
                'visualization': False
            },
            
            'top_analyse': {
                'keywords': ['top', 'meilleur', 'premier', 'classement', 'ranking'],
                'validation': lambda df: True,
                'code_template': """
# Extraction du nombre et de la colonne à analyser
import re
question_lower = "{question}".lower()
numbers = re.findall(r'\\d+', "{question}")
n = int(numbers[0]) if numbers else 10

# Détection de la colonne à analyser
if any(word in question_lower for word in ['navire', 'bateau']):
    if 'NOM_NAVIRE' not in df.columns:
        raise ValueError("Colonne NOM_NAVIRE introuvable")
    col = 'NOM_NAVIRE'
    title = f"Top {n} des Navires par Nombre d'Escales"
elif any(word in question_lower for word in ['armateur']):
    if 'LIBELLE_ARMATEUR' not in df.columns:
        raise ValueError("Colonne LIBELLE_ARMATEUR introuvable")
    col = 'LIBELLE_ARMATEUR'
    title = f"Top {n} des Armateurs par Nombre d'Escales"
elif any(word in question_lower for word in ['type']):
    if 'TYPE_NAVIRE' not in df.columns:
        raise ValueError("Colonne TYPE_NAVIRE introuvable")
    col = 'TYPE_NAVIRE'
    title = f"Top {n} des Types de Navires"
elif any(word in question_lower for word in ['pavillon', 'pays']):
    if 'PAVILLON' not in df.columns:
        raise ValueError("Colonne PAVILLON introuvable")
    col = 'PAVILLON'
    title = f"Top {n} des Pavillons"
else:
    col = 'NOM_NAVIRE'  # Par défaut
    title = f"Top {n} des Navires par Nombre d'Escales"

result = df[col].value_counts().head(n)
if result.empty:
    raise ValueError(f"Aucune donnée trouvée pour la colonne {col}")

print(f"Analyse: {title}")
print(f"Total d'entrées uniques: {df[col].nunique()}")
""",
                'visualization': True
            },
            
            'statistiques_generales': {
                'keywords': [
                    'statistique', 'statistiques', 'statistiques générales', 'statistiques du port',
                    'résumé', 'synthèse', 'bilan', 'global', 'général', 'overview',
                    'indicateurs', 'chiffres clés', 'bilan global', 'bilan portuaire', 'summary', 'general stats', 'general statistics', 'key figures', 'synthese', 'recap', 'recapitulatif', 'stat global', 'stat port', 'statistiques portuaires'
                ],
                'validation': lambda df: True,
                'code_template': """
# Statistiques générales complètes
stats = {}

# Informations de base
stats['total_escales'] = len(df)
stats['periode'] = {
    'debut': df['DATETIME_RADE'].min() if 'DATETIME_RADE' in df.columns else 'N/A',
    'fin': df['DATETIME_RADE'].max() if 'DATETIME_RADE' in df.columns else 'N/A'
}

# Types de navires
if 'TYPE_NAVIRE' in df.columns:
    stats['types_navires'] = df['TYPE_NAVIRE'].value_counts().to_dict()

# Pavillons
if 'PAVILLON' in df.columns:
    stats['top_pavillons'] = df['PAVILLON'].value_counts().head(5).to_dict()

# Tonnages
if 'TONNAGE_IMPORT' in df.columns:
    stats['tonnage_import_total'] = df['TONNAGE_IMPORT'].sum()
if 'TONNAGE_EXPORT' in df.columns:
    stats['tonnage_export_total'] = df['TONNAGE_EXPORT'].sum()

# Temps d'attente corrigé
if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in df.columns:
    temps_data = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
    temps_data = temps_data[(temps_data >= 0) & (temps_data <= 240)]
    if len(temps_data) > 0:
        stats['temps_attente_moyen'] = temps_data.mean()
        stats['temps_attente_mediane'] = temps_data.median()

print("=== STATISTIQUES GÉNÉRALES ===")
print(f"Total des escales: {stats['total_escales']}")
print(f"Période: {stats['periode']['debut']} à {stats['periode']['fin']}")

if 'types_navires' in stats:
    print("\\nTypes de navires:")
    for type_nav, count in list(stats['types_navires'].items())[:5]:
        print(f"  - {type_nav}: {count}")

if 'top_pavillons' in stats:
    print("\\nTop 5 des pavillons:")
    for pavillon, count in stats['top_pavillons'].items():
        print(f"  - {pavillon}: {count}")

if 'tonnage_import_total' in stats:
    print(f"\\nTonnage import total: {stats['tonnage_import_total']:,.0f} tonnes")
if 'tonnage_export_total' in stats:
    print(f"Tonnage export total: {stats['tonnage_export_total']:,.0f} tonnes")

if 'temps_attente_moyen' in stats:
    print(f"\\nTemps d'attente moyen (corrigé): {stats['temps_attente_moyen']:.2f} heures")
    print(f"Temps d'attente médiane: {stats['temps_attente_mediane']:.2f} heures")

result = stats
""",
                'visualization': False
            }
        }
        return templates
    
    def load_and_prepare_data(self):
        """Load and comprehensively prepare data"""
        try:
            print("📄 Chargement et préparation des données...")
            self.df_original = pd.read_csv(self.csv_path)
            print(f"✅ CSV chargé: {self.df_original.shape[0]} lignes, {self.df_original.shape[1]} colonnes")
            
            # Apply enhanced data preparation
            self.df = self._enhanced_data_preparation(self.df_original.copy())
            
            print(f"✅ Données préparées: {len(self.df)} lignes, {len(self.df.columns)} colonnes")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            sys.exit(1)
    
    def _enhanced_data_preparation(self, df):
        """Enhanced data preparation with correct complex date handling"""
        
        print("🔧 Traitement des dates et heures complexes...")
        
        # Traitement correct des dates complexes
        df = self._process_complex_dates_and_times(df)
        
        # Création des colonnes DateTime complètes
        df = self._create_complete_datetime_columns(df)
        
        # Calcul des durées corrigées
        df = self._calculate_corrected_durations(df)
        
        # Validation de la qualité des données
        self._validate_data_quality(df)
        
        return df
    
    def _process_complex_dates_and_times(self, df):
        """Traitement correct des dates et heures avec séparateurs |"""
        
        date_cols = ['DATE_RADE', 'DATE_ENTREE_PORT', 'DATE_ACCOSTAGE', 'DATE_APP_QUAI', 'DATE_APP_PORT']
        heure_cols = ['HEURE_RADE', 'HEURE_ENTREE_PORT', 'HEURE_ACCOSTAGE', 'HEURE_APP_QUAI', 'HEURE_APP_PORT']
        
        # Traitement des dates avec logique de sélection de la dernière date si multiples
        for col in date_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_complex_dates)
                valid_count = df[col].notna().sum()
                print(f"   📅 {col}: {valid_count}/{len(df)} dates valides")
        
        # Traitement des heures avec logique de sélection de la dernière heure si multiples  
        for col in heure_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._parse_complex_times)
                valid_count = df[col].notna().sum()
                print(f"   ⏰ {col}: {valid_count}/{len(df)} heures valides")
        
        return df

    def _parse_complex_dates(self, date_str):
        """Parse dates complexes - retourne la dernière date si multiples"""
        if pd.isna(date_str) or str(date_str).strip() == '':
            return pd.NaT

        date_str = str(date_str).strip()
        formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y']

        if '|' in date_str:
            dates = [s.strip() for s in date_str.split('|')]
            parsed_dates = []
            for part in dates:
                parsed = pd.NaT
                for fmt in formats:
                    try:
                        parsed = pd.to_datetime(part, format=fmt, errors='coerce')
                        if not pd.isna(parsed):
                            break
                    except Exception:
                        pass
                if pd.isna(parsed):
                    try:
                        parsed = pd.to_datetime(part, dayfirst=True, errors='coerce')
                    except Exception:
                        parsed = pd.NaT
                if not pd.isna(parsed):
                    parsed_dates.append(parsed)
            return parsed_dates[-1] if parsed_dates else pd.NaT
        else:
            # Une seule date
            parsed = pd.NaT
            for fmt in formats:
                try:
                    parsed = pd.to_datetime(date_str, format=fmt, errors='coerce')
                    if not pd.isna(parsed):
                        break
                except Exception:
                    pass
            if pd.isna(parsed):
                try:
                    parsed = pd.to_datetime(date_str, dayfirst=True, errors='coerce')
                except Exception:
                    parsed = pd.NaT
            return parsed
    
    def _parse_complex_times(self, time_str):
        """Parse times complexes - retourne la dernière heure si multiples"""
        if pd.isna(time_str) or str(time_str).strip() == '':
            return None

        time_str = str(time_str).strip()
        formats = ['%H:%M', '%H:%M:%S', '%H.%M']

        if '|' in time_str:
            times = [s.strip() for s in time_str.split('|')]
            parsed_times = []
            for part in times:
                parsed_time = None
                for fmt in formats:
                    try:
                        tmp = pd.to_datetime(part, format=fmt, errors='coerce')
                        if not pd.isna(tmp):
                            parsed_time = tmp.time()
                            break
                    except Exception:
                        pass
                if parsed_time is not None:
                    parsed_times.append(parsed_time)
            return parsed_times[-1] if parsed_times else None
        else:
            parsed_time = None
            for fmt in formats:
                try:
                    tmp = pd.to_datetime(time_str, format=fmt, errors='coerce')
                    if not pd.isna(tmp):
                        parsed_time = tmp.time()
                        break
                except Exception:
                    continue
            return parsed_time
    
    def _create_complete_datetime_columns(self, df):
        """Création des colonnes DateTime complètes en combinant dates et heures"""
        
        print("🔗 Combinaison des dates et heures...")
        
        date_heure_pairs = [
            ('DATE_RADE', 'HEURE_RADE', 'DATETIME_RADE'),
            ('DATE_ENTREE_PORT', 'HEURE_ENTREE_PORT', 'DATETIME_ENTREE_PORT'),
            ('DATE_ACCOSTAGE', 'HEURE_ACCOSTAGE', 'DATETIME_ACCOSTAGE'),
            ('DATE_APP_QUAI', 'HEURE_APP_QUAI', 'DATETIME_APP_QUAI'),
            ('DATE_APP_PORT', 'HEURE_APP_PORT', 'DATETIME_APP_PORT')
        ]
        
        for date_col, heure_col, datetime_col in date_heure_pairs:
            if date_col in df.columns and heure_col in df.columns:
                df[datetime_col] = df.apply(
                    lambda row: self._combine_date_time_safe(row[date_col], row[heure_col]),
                    axis=1
                )
                valid_count = df[datetime_col].notna().sum()
                print(f"   🕕 {datetime_col}: {valid_count}/{len(df)} datetime valides")
        
        return df
    
    def _combine_date_time_safe(self, date_val, time_val):
        """Combine date et heure de manière sécurisée"""
        if pd.isnull(date_val) or pd.isnull(time_val):
            return pd.NaT
        
        try:
            if isinstance(date_val, pd.Timestamp):
                return pd.Timestamp.combine(date_val.date(), time_val)
            else:
                return pd.NaT
        except Exception:
            return pd.NaT
    
    def _calculate_corrected_durations(self, df):
        """Calcul des durées corrigées avec validation"""
        
        print("🧮 Calcul des durées corrigées...")
        
        # Temps d'attente entre arrivée en rade et accostage (CORRIGÉ)
        if 'DATETIME_RADE' in df.columns and 'DATETIME_ACCOSTAGE' in df.columns:
            mask = df['DATETIME_RADE'].notna() & df['DATETIME_ACCOSTAGE'].notna()
            df.loc[mask, 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'] = (
                df.loc[mask, 'DATETIME_ACCOSTAGE'] - df.loc[mask, 'DATETIME_RADE']
            ).dt.total_seconds() / 3600

            valid_count = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].notna().sum()
            print(f"   ⏱️  TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE: {valid_count} valeurs calculées")
            
            if valid_count > 0:
                temps_data = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
                print(f"   📊 Temps d'attente: min={temps_data.min():.2f}h, max={temps_data.max():.2f}h, moy={temps_data.mean():.2f}h")
        
        # Temps entre entrée au port et accostage (CORRIGÉ)
        if 'DATETIME_ENTREE_PORT' in df.columns and 'DATETIME_ACCOSTAGE' in df.columns:
            mask = df['DATETIME_ENTREE_PORT'].notna() & df['DATETIME_ACCOSTAGE'].notna()
            df.loc[mask, 'TEMPS_ENTREE_ACCOSTAGE_HEURES_CORRIGE'] = (
                df.loc[mask, 'DATETIME_ACCOSTAGE'] - df.loc[mask, 'DATETIME_ENTREE_PORT']
            ).dt.total_seconds() / 3600
            print(f"   ⏱️  TEMPS_ENTREE_ACCOSTAGE_HEURES_CORRIGE: {df['TEMPS_ENTREE_ACCOSTAGE_HEURES_CORRIGE'].notna().sum()} valeurs calculées")
        
        # Temps de séjour au quai (CORRIGÉ)
        if 'DATETIME_ACCOSTAGE' in df.columns and 'DATETIME_APP_QUAI' in df.columns:
            mask = df['DATETIME_ACCOSTAGE'].notna() & df['DATETIME_APP_QUAI'].notna()
            df.loc[mask, 'TEMPS_SEJOUR_QUAI_HEURES_CORRIGE'] = (
                df.loc[mask, 'DATETIME_APP_QUAI'] - df.loc[mask, 'DATETIME_ACCOSTAGE']
            ).dt.total_seconds() / 3600
            print(f"   ⏱️  TEMPS_SEJOUR_QUAI_HEURES_CORRIGE: {df['TEMPS_SEJOUR_QUAI_HEURES_CORRIGE'].notna().sum()} valeurs calculées")
        
        return df
    
    def _validate_data_quality(self, df):
        """Validation avancée de la qualité des données"""
        print("🔍 Validation de la qualité des données...")
        
        issues = []
        
        # Vérification des colonnes critiques
        critical_cols = ['NOM_NAVIRE', 'TYPE_NAVIRE', 'DATETIME_RADE']
        for col in critical_cols:
            if col not in df.columns:
                issues.append(f"Colonne critique manquante: {col}")
            elif df[col].isna().sum() > len(df) * 0.5:
                issues.append(f"Plus de 50% de valeurs manquantes dans {col}")
        
        # Validation des temps d'attente corrigés
        if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in df.columns:
            temps_data = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
            negative_wait = (temps_data < 0).sum()
            extreme_wait = (temps_data > 720).sum()  # Plus de 30 jours
            
            if negative_wait > 0:
                issues.append(f"Temps d'attente négatifs: {negative_wait} cas")
            if extreme_wait > 0:
                issues.append(f"Temps d'attente extrêmes (>30j): {extreme_wait} cas")
        
        # Validation des longueurs de navire
        if 'LONGEUR_NAVIRE' in df.columns:
            invalid_length = ((df['LONGEUR_NAVIRE'] < 0) | (df['LONGEUR_NAVIRE'] > 500)).sum()
            if invalid_length > 0:
                issues.append(f"LONGEUR_NAVIRE: {invalid_length} valeurs aberrantes")
        
        if issues:
            print("⚠️  Problèmes détectés:")
            for issue in issues:
                print(f"   - {issue}")
            self.error_history.extend(issues)
        else:
            print("✅ Qualité des données validée")
        
        # Statistiques de comparaison
        if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES' in df.columns and 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in df.columns:
            ancien = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES'].dropna().mean()
            nouveau = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna().mean()
            print(f"📊 Comparaison temps d'attente:")
            print(f"   - Ancien calcul: {ancien:.2f}h")
            print(f"   - Nouveau calcul (corrigé): {nouveau:.2f}h")
            print(f"   - Différence: {abs(ancien-nouveau):.2f}h")
    
    def intelligent_query_processing(self, question: str) -> Dict[str, Any]:
        """Process query using hybrid approach: templates first, then LLM"""
        
        start_time = time.time()
        self.performance_stats["total_queries"] += 1
        
        print(f"\n🤖 Traitement intelligent: {question}")
        
        # Step 1: Try template matching
        template_result = self._try_template_analysis(question)
        
        if template_result['success']:
            execution_time = time.time() - start_time
            self.performance_stats["successful_queries"] += 1
            self._update_avg_response_time(execution_time)
            
            print(f"✅ Analyse par template en {execution_time:.2f}s")
            return template_result
        
        # Step 2: Fallback to LLM generation with enhanced validation
        print("🔄 Basculement vers génération LLM...")
        llm_result = self._llm_analysis_with_validation(question)
        
        execution_time = time.time() - start_time
        if llm_result['success']:
            self.performance_stats["successful_queries"] += 1
        self._update_avg_response_time(execution_time)
        
        print(f"⏱️  Temps total: {execution_time:.2f}s")
        return llm_result
    
    def _try_template_analysis(self, question: str) -> Dict[str, Any]:
        """Try to match question with predefined templates, with stricter logic for temps d'attente moyen."""
        question_lower = question.lower()

        # Détection stricte de la question sur le temps d'attente moyen entre rade et accostage
        attente_patterns = [
            "temps d’attente moyen entre l’arrivée en rade et l’accostage au quai",
            "temps d'attente moyen entre l'arrivée en rade et l'accostage au quai",
            "temps d’attente moyen entre l’arrivée en rade et l’accostage",
            "temps d'attente moyen entre l'arrivée en rade et l'accostage",
            "temps d’attente moyen en rade",
            "temps d'attente moyen en rade",
            "délai moyen en rade",
            "délai moyen entre l’arrivée en rade et l’accostage au quai",
            "délai moyen entre l'arrivée en rade et l'accostage au quai"
        ]
        for pat in attente_patterns:
            if pat in question_lower:
                # Calcul direct, bypass template system
                try:
                    if 'DATETIME_RADE' in self.df.columns and 'DATETIME_ACCOSTAGE' in self.df.columns:
                        df_valid = self.df.dropna(subset=['DATETIME_RADE', 'DATETIME_ACCOSTAGE']).copy()
                        df_valid['DATETIME_RADE'] = pd.to_datetime(df_valid['DATETIME_RADE'], errors='coerce')
                        df_valid['DATETIME_ACCOSTAGE'] = pd.to_datetime(df_valid['DATETIME_ACCOSTAGE'], errors='coerce')
                        attente = (df_valid['DATETIME_ACCOSTAGE'] - df_valid['DATETIME_RADE']).dt.total_seconds() / 3600
                        attente_filtrée = attente[(attente >= 0) & (attente <= 240)]
                        moyenne = attente_filtrée.mean()
                        nb_extremes = (attente > 720).sum()
                        nb_lignes = len(self.df)
                        nb_colonnes = len(self.df.columns)
                        msg = ""
                        if nb_extremes > 0:
                            msg += "Problèmes détectés:\n   - Temps d'attente extrêmes (>30j): {} cas\n".format(nb_extremes)
                        msg += "✅ Données préparées: {} lignes, {} colonnes\n".format(nb_lignes, nb_colonnes)
                        msg += "Temps d'attente moyen entre l'arrivée en rade et l'accostage au quai : {:.2f} heures".format(moyenne if moyenne is not None else 0)
                        return {
                            'success': True,
                            'result': msg,
                            'method': 'direct',
                            'template_used': 'temps_attente_direct',
                            'error': None
                        }
                    else:
                        return {'success': False, 'error': 'Colonnes nécessaires manquantes dans le DataFrame.'}
                except Exception as e:
                    return {'success': False, 'error': f'Erreur de calcul du temps d\'attente: {str(e)}'}

        # Sinon, matching template classique, mais priorité au template temps_attente
        matched_template = None
        # Priorité au template temps_attente si tous les mots-clés sont présents
        temps_attente_keywords = self.analysis_templates['temps_attente']['keywords']
        if all(kw in question_lower for kw in temps_attente_keywords):
            if self.analysis_templates['temps_attente']['validation'](self.df):
                matched_template = 'temps_attente'
        if not matched_template:
            # Sinon, matching classique, mais on ignore productivite_terminaux si question contient 'attente' ou 'moyenne'
            for template_name, template_info in self.analysis_templates.items():
                if template_name == 'productivite_terminaux' and (
                    'attente' in question_lower or 'moyenne' in question_lower):
                    continue
                if any(keyword in question_lower for keyword in template_info['keywords']):
                    if template_info['validation'](self.df):
                        matched_template = template_name
                        break
        if not matched_template:
            return {'success': False, 'error': 'Aucun template correspondant trouvé'}
        try:
            # Execute template
            template_info = self.analysis_templates[matched_template]
            code = template_info['code_template'].format(question=question)
            # Safe execution
            local_vars = {"df": self.df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
            exec(code, {}, local_vars)
            result = local_vars.get("result", "Analyse terminée")
            return {
                'success': True,
                'result': result,
                'method': 'template',
                'template_used': matched_template,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Erreur template {matched_template}: {str(e)}",
                'method': 'template_failed'
            }
    
    def _llm_analysis_with_validation(self, question: str) -> Dict[str, Any]:
        """Enhanced LLM analysis with validation and forced plot if requested."""
        # Generate enhanced prompt
        prompt = self._generate_enhanced_prompt(question)
        try:
            # Call Ollama
            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1, "top_p": 0.9}
            )
            # Clean and validate code
            raw_code = response['message']['content']
            cleaned_code = self._clean_and_validate_code(raw_code, question)
            # Forçage du graphique si demandé dans la question
            question_lower = question.lower()
            plot_words = ["graph", "graphique", "plot", "courbe", "visualiser"]
            if any(w in question_lower for w in plot_words):
                code_lines = cleaned_code.splitlines()
                has_plot = any("plt.show()" in l or ".plot(" in l for l in code_lines)
                if not has_plot:
                    # Inject plot code for result if it's a Series or DataFrame
                    plot_code = "\nimport matplotlib.pyplot as plt\nif isinstance(result, (pd.Series, pd.DataFrame)):\n    result.plot(kind='bar', figsize=(12,6))\n    plt.title(\"Graphique demandé par l'utilisateur\")\n    plt.tight_layout()\n    plt.show()"
                    cleaned_code = cleaned_code.rstrip() + plot_code
            print(f"🔧 Code généré:\n{cleaned_code}")
            # Execute with enhanced error handling
            result, error = self._execute_with_advanced_retry(cleaned_code)
            if error is None:
                return {
                    'success': True,
                    'result': result,
                    'method': 'llm',
                    'code_generated': cleaned_code,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'error': error,
                    'method': 'llm_failed',
                    'code_attempted': cleaned_code
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Erreur LLM: {str(e)}",
                'method': 'llm_error'
            }
    
    def _generate_enhanced_prompt(self, question: str) -> str:
        """Generate enhanced prompt with context and constraints"""
        
        # Get available calculated columns
        calculated_cols = [col for col in self.df.columns if col.startswith('TEMPS_') or col.startswith('DATETIME_')]
        
        # Generate smart suggestions based on question
        suggestions = self._generate_smart_suggestions(question)
        
        # Error context
        error_context = ""
        if self.error_history:
            recent_errors = self.error_history[-3:]
            error_context = f"\nErreurs récentes à éviter: {recent_errors}"
        
        return f"""
{self.column_descriptions}

Colonnes disponibles: {', '.join(self.df.columns)}

Colonnes pré-calculées CORRIGÉES: {', '.join(calculated_cols)}

RÈGLES CRITIQUES OPTIMISÉES:
1. PRIORITÉ AUX COLONNES CORRIGÉES: Utiliser TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE au lieu de TEMPS_ATTENTE_ACCOSTAGE_HEURES
2. UTILISE EXACTEMENT les noms de colonnes listés
3. Pour temps d'attente CORRIGÉ: df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].mean()
4. Pour analyses temporelles: utiliser DATETIME_RADE, DATETIME_ACCOSTAGE
5. Pour top N: df['COLONNE'].value_counts().head(N)
6. TOUJOURS assigner le résultat à 'result'
7. Gestion des valeurs manquantes: .dropna() avant calculs
8. Filtrage des valeurs aberrantes pour temps: between(0, 240) pour heures
9. Validation: vérifier len(data) > 0 avant opérations
10. **IMPORTANT**: Ne jamais utiliser pd.read_csv() - le DataFrame 'df' est déjà disponible
11. **CRITIQUE**: Utiliser SEULEMENT le DataFrame 'df' fourni, ne pas charger de nouveau fichier

{suggestions}

EXEMPLES VALIDÉS CORRIGÉS:
- Temps moyen CORRIGÉ: result = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna().mean()
- Top navires: result = df['NOM_NAVIRE'].value_counts().head(10)
- Évolution: result = df.groupby(df['DATETIME_RADE'].dt.year).size() si DATETIME_RADE
- Stats par type: result = df.groupby('TYPE_NAVIRE')['TONNAGE_IMPORT'].sum()
- Coûts remorquage: result = df[df['NBR_REMORQUEUR_ACCOSTAGE'] > 0]['MONTANT_HT_FACTURE_DM'].sum()

IMPORTANT: 
- Les colonnes DATETIME_* contiennent les dates/heures complètes correctement combinées.
- Les colonnes TEMPS_*_CORRIGE contiennent les durées calculées correctement.
- Le DataFrame 'df' est DÉJÀ CHARGÉ et prêt à utiliser
- NE JAMAIS utiliser pd.read_csv() ou charger un fichier

{error_context}

Question: {question}
Code Python (uniquement du code exécutable, utilisant le DataFrame 'df' déjà fourni):"""
    
    def _generate_smart_suggestions(self, question: str) -> str:
        """Generate smart suggestions based on question analysis"""
        question_lower = question.lower()
        suggestions = []
        
        # Temporal analysis suggestions
        if any(word in question_lower for word in ['évolution', 'tendance', 'temporel']):
            if 'DATETIME_RADE' in self.df.columns:
                suggestions.append("💡 Pour évolution temporelle: df.groupby(df['DATETIME_RADE'].dt.year).size()")
        
        # Wait time suggestions - CORRIGÉ
        if any(word in question_lower for word in ['attente', 'délai', 'temps']):
            if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in self.df.columns:
                suggestions.append("💡 Temps d'attente CORRIGÉ disponible: df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].mean()")
        
        # Top analysis suggestions
        if any(word in question_lower for word in ['top', 'meilleur', 'premier']):
            suggestions.append("💡 Pour classements: df['COLONNE'].value_counts().head(N)")
        
        # Visualization suggestions
        if any(word in question_lower for word in ['graphique', 'courbe', 'plot', 'visualiser']):
            suggestions.append("💡 Pour visualiser: import matplotlib.pyplot as plt; df.plot(...); plt.show()")
        
        return "\n".join(suggestions) if suggestions else ""
    
    def _clean_and_validate_code(self, code: str, question: str = "") -> str:
        # Parsing automatique de la question pour extraire entité, métrique, agrégation, condition
        import re
        question_lower = question.lower() if question else ""
        op_map = [
            (r"moins de (\d+)", "<"),
            (r"plus de (\d+)", ">"),
            (r"au moins (\d+)", ">="),
            (r"au plus (\d+)", "<="),
            (r"supérieur[e]? à (\d+)", ">"),
            (r"inférieur[e]? à (\d+)", "<"),
            (r"égal(?:e)? à (\d+)", "=="),
            (r"exactement (\d+)", "==")
        ]
        entity_map = {
            'navire': 'NOM_NAVIRE',
            'navires': 'NOM_NAVIRE',
            'armateur': 'LIBELLE_ARMATEUR',
            'armateurs': 'LIBELLE_ARMATEUR',
            'pavillon': 'PAVILLON',
            'pavillons': 'PAVILLON',
            'type': 'TYPE_NAVIRE',
            'types': 'TYPE_NAVIRE',
            'destination': 'DESTINATION',
            'provenance': 'PROVENANCE',
            'année': 'ANNEE',
            'annee': 'ANNEE',
            'annees': 'ANNEE',
        }
        metric_map = [
            (['escale', 'escales'], 'NUMERO_ESCALE', 'count'),
            (['tonnage import', 'tonnage_import'], 'TONNAGE_IMPORT', 'sum'),
            (['tonnage export', 'tonnage_export'], 'TONNAGE_EXPORT', 'sum'),
            (['remorqueur', 'remorqueurs'], 'NBR_REMORQUEUR_ACCOSTAGE', 'sum'),
            (['moyenne temps attente', 'temps attente moyen', 'temps d\'attente moyen'], 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE', 'mean'),
            (['moyenne', 'mean'], None, 'mean'),
        ]
        # Extraction entité
        entity = None
        for key in entity_map:
            if key in question_lower:
                entity = entity_map[key]
                break
        # Extraction métrique et agrégation
        metric_col = 'NUMERO_ESCALE'
        agg = 'count'
        for keywords, col, agg_type in metric_map:
            if any(k in question_lower for k in keywords):
                if col:
                    metric_col = col
                agg = agg_type
                break
        # Extraction top N
        n = None
        n_match = re.search(r"donne[rz]?\s*(moi)?\s*(\d+)\s+" + (key if entity else "") + r"", question_lower)
        if n_match:
            n = int(n_match.group(2))
        # Extraction opérateur et seuil
        op = None
        seuil = None
        for pattern, op_str in op_map:
            match = re.search(pattern, question_lower)
            if match:
                op = op_str
                seuil = int(match.group(1))
                break
        # Si question complexe (entité, métrique, agrégation, condition), applique le template adapté
        if entity and metric_col and agg and op and seuil is not None:
            if agg == 'count':
                code_joined = (
                    f"result = df.groupby('{entity}')['{metric_col}'].count()\n"
                    f"result = result[result {op} {seuil}]\n"
                    + (f"result = result.head({n})" if n else "")
                )
            elif agg == 'sum':
                code_joined = (
                    f"result = df.groupby('{entity}')['{metric_col}'].sum()\n"
                    f"result = result[result {op} {seuil}]\n"
                    + (f"result = result.head({n})" if n else "")
                )
            elif agg == 'mean':
                code_joined = (
                    f"result = df.groupby('{entity}')['{metric_col}'].mean()\n"
                    f"result = result[result {op} {seuil}]\n"
                    + (f"result = result.head({n})" if n else "")
                )
            else:
                code_joined = (
                    f"result = df.groupby('{entity}')['{metric_col}'].{agg}()\n"
                    f"result = result[result {op} {seuil}]\n"
                    + (f"result = result.head({n})" if n else "")
                )
            return code_joined
        # Sinon, laisse le LLM générer, mais vérifie la présence de groupby, agrégation et filtrage
        # Si le code LLM ne contient pas ces éléments, tente de corriger automatiquement
        code_lower = code.lower()
        if entity and metric_col and agg and op and seuil is not None:
            if not (f"groupby('{entity.lower()}'" in code_lower and f"{agg}(" in code_lower and op in code_lower):
                # Correction automatique
                if agg == 'count':
                    code_joined = (
                        f"result = df.groupby('{entity}')['{metric_col}'].count()\n"
                        f"result = result[result {op} {seuil}]\n"
                        + (f"result = result.head({n})" if n else "")
                    )
                elif agg == 'sum':
                    code_joined = (
                        f"result = df.groupby('{entity}')['{metric_col}'].sum()\n"
                        f"result = result[result {op} {seuil}]\n"
                        + (f"result = result.head({n})" if n else "")
                    )
                elif agg == 'mean':
                    code_joined = (
                        f"result = df.groupby('{entity}')['{metric_col}'].mean()\n"
                        f"result = result[result {op} {seuil}]\n"
                        + (f"result = result.head({n})" if n else "")
                    )
                else:
                    code_joined = (
                        f"result = df.groupby('{entity}')['{metric_col}'].{agg}()\n"
                        f"result = result[result {op} {seuil}]\n"
                        + (f"result = result.head({n})" if n else "")
                    )
                return code_joined
              # DÉBUT DE LA CORRECTION POUR LE PROBLÈME DE SYNTAXE
    # Cas spécial pour graphiques TOP demandés explicitement
 
            if any(word in question_lower for word in ["graph", "graphique", "plot", "courbe", "visualiser", "top"]):
                # Extraire le nombre (par défaut 10)
                numbers = re.findall(r'\d+', question)
                n = int(numbers[0]) if numbers else 10

                print("[LOG] Génération d'un graphique pour top des navires")
                 
                # Code spécialisé pour top navires avec graphique
                specialized_code = f"""import matplotlib.pyplot as plt
    result = df['NOM_NAVIRE'].value_counts().head({n})
    plt.figure(figsize=(12, 8))
    result.plot(kind='bar')
    plt.title('Top {n} des Navires par Nombre d\\'Escales')
    plt.xlabel('Nom du Navire')
    plt.ylabel('Nombre d\\'Escales')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()"""
            
                return specialized_code.strip()
    # FIN DE LA CORRECTION SPÉCIALISÉE

        # Sinon, retourne le code LLM tel quel
        # Enhanced code cleaning and validation
        import re
        # Remove markdown and comments
        code = re.sub(r"```[\w]*", "", code)
        code = code.replace("```", "").replace("`", "")

        # Remove all print statements and keep only valid Python code lines
        lines = []
        for line in code.splitlines():
            line = line.strip()
            # Ignore empty lines, comments, markdown, and explanations
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            if line.startswith("'''") or line.startswith('"""'):
                continue
            # NOUVEAU : Ignorer TOUTES les phrases explicatives en anglais/français
            if (line.startswith("Here is") or line.startswith("This code") or 
                line.startswith("Note that") or line.startswith("The resulting") or 
                line.startswith("If your data") or line.startswith("The code") or
                line.startswith("This will") or line.startswith("You can") or
                line.startswith("Make sure") or line.startswith("Remember that") or
                line.startswith("Voici le") or line.startswith("Ce code") or
                line.startswith("Notez que")):
               continue
            # NOUVEAU : Ignorer les lignes contenant des mots explicatifs
            if any(skip in line.lower() for skip in [
              "here is", "this code", "the output", "will be", "creates", 
              "markdown", "resulting series", "note that", "if your data", 
              "python code", "uses the", "method to", "and then", 
              "assumes that", "structured differently", "may need to modify"
            ]):
               continue

            if "pd.read_csv" in line or "read_csv" in line:
                continue
            if line.startswith("print(") or line.startswith("print "):
                continue
                    # NOUVEAU : Ne garder QUE les lignes qui ressemblent vraiment à du code Python
            if (re.match(r"^[\w\[\]\.]+ ?= ?.+", line) or  # Assignations
                re.match(r"^[\w\.]+\(.+\)$", line) or      # Appels de fonction  
                line.startswith("import ") or              # Imports
                line.startswith("plt.") or                 # Matplotlib
                line.startswith("df") or                   # DataFrame operations
               "result =" in line):                       # Assignation result
                lines.append(line)


        # Cas spécial : question de comparaison import/export par pays
        question_lower = question.lower() if question else ""
        if ("import" in question_lower and "export" in question_lower and "pays" in question_lower):
            print("[LOG] Utilisation du code spécial comparaison import/export par pays (bypass LLM)")
            code_joined = (
                "imported_by_country = df.groupby('PAYS_PROVENANCE')['TONNAGE_MARCHANDISES_ESCALE_DAP_IMPORT'].sum()\n"
                "exported_by_country = df.groupby('PAYS_DESTINATION')['TONNAGE_MARCHANDISES_ESCALE_DAP_EXPORT'].sum()\n"
                "all_countries = set(imported_by_country.index).union(set(exported_by_country.index))\n"
                "result_df = pd.DataFrame({'Import': imported_by_country, 'Export': exported_by_country}).fillna(0)\n"
                "result_df = result_df.loc[all_countries] if hasattr(result_df, 'loc') else result_df\n"
                "result_df['Total'] = result_df['Import'] + result_df['Export']\n"
                "result_df = result_df.sort_values('Total', ascending=False).head(15)\n"
                "result = result_df\n"
                "import matplotlib.pyplot as plt\n"
                "result_df[['Import', 'Export']].plot(kind='bar', figsize=(12,6))\n"
                "plt.title('Comparaison Import/Export par Pays (Top 15)')\n"
                "plt.ylabel('Tonnage')\n"
                "plt.xlabel('Pays')\n"
                "plt.tight_layout()\n"
                "plt.show()"
            )
            return code_joined

        # Cas spécial : insistance sur un graphique
        if any(word in question_lower for word in ["graph", "graphique", "plot", "courbe", "visualiser"]):
            print("[LOG] Forçage d'un graphique car la question le demande explicitement.")
            code_lines = code.splitlines()
            has_plot = any("plt.show()" in l or ".plot(" in l for l in code_lines)
            plot_code = "\nimport matplotlib.pyplot as plt\nif isinstance(result, (pd.Series, pd.DataFrame)):\n    result.plot(kind='bar', figsize=(12,6))\n    plt.title(\"Graphique demandé par l'utilisateur\")\n    plt.tight_layout()\n    plt.show()"
            if not has_plot:
                code = code.rstrip() + plot_code
            return code
        else:
            # Cas spécial : question sur le nombre d'escales dans un port donné
            if ("nombre" in question_lower or "combien" in question_lower) and ("escale" in question_lower or "escales" in question_lower) and ("port" in question_lower):
                port_match = re.search(r"port(?: de| du| d'| des)? ([a-zA-Zéèêàâîôûç\-']+)", question_lower)
                port_name = port_match.group(1).strip().capitalize() if port_match else None
                port_columns = ['PORT', 'PORT_ESCALE', 'PORT_DU_NAVIRE', 'PORT_DE_DESTINATION', 'DESTINATION', 'PROVENANCE']
                code_lines = []
                if port_name:
                    code_lines.append("# Filtrer les escales pour le port demandé")
                    code_lines.append("port_col = None")
                    code_lines.append("for col in ['PORT', 'PORT_ESCALE', 'PORT_DU_NAVIRE', 'PORT_DE_DESTINATION', 'DESTINATION', 'PROVENANCE']:")
                    code_lines.append("    if col in df.columns:")
                    code_lines.append(f"        if df[col].astype(str).str.lower().str.contains('{port_name.lower()}').any():")
                    code_lines.append("            port_col = col")
                    code_lines.append("            break")
                    code_lines.append("if port_col:")
                    code_lines.append(f"    result = df[df[port_col].astype(str).str.lower().str.contains('{port_name.lower()}')].shape[0]")
                    code_lines.append("else:")
                    code_lines.append("    result = 0  # Colonne de port non trouvée")
                    code_joined = "\n".join(code_lines)
                else:
                    code_joined = "result = 0  # Port non détecté dans la question"
            else:
                assign_lines = []
                for l in lines:
                    # Ignore multiple assignments like 'result = grouped_df = ...'
                    if re.match(r"^result\s*=\s*[\w_]+\s*=", l):
                        # Only keep the rightmost assignment for result
                        right = l.split('=')[-1].strip()
                        assign_lines.append(f"result = {right}")
                    elif l.count('=') == 1 and re.match(r"^[\w\[\]\.]+ ?= ?.+", l):
                        assign_lines.append(l)
                result_lines = [l for l in assign_lines if l.strip().startswith("result =")]
                plot_code = ""
                if any(word in question_lower for word in ["graphique", "courbe", "plot", "visualiser"]):
                    plot_code = "\nimport matplotlib.pyplot as plt\nif isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):\n    result.plot(kind='bar')\n    plt.title('Analyse des données portuaires')\n    plt.ylabel('Valeur')\n    plt.xlabel('Catégorie')\n    plt.tight_layout()\n    plt.show()"
                if result_lines:
                    idx = assign_lines.index(result_lines[-1])
                    code_joined = "\n".join(assign_lines[:idx+1]) + plot_code
                elif assign_lines:
                    code_joined = "\n".join(assign_lines)
                    last = assign_lines[-1]
                    if not last.startswith("result"):
                        code_joined += f"\nresult = {last.split('=')[0].strip()}"
                    code_joined += plot_code
                else:
                    raise RuntimeError("Aucun code Python valide généré par le LLM. Reformulez la question.")

        # Apply column mapping corrections
        for term, actual_col in self.column_mapping.items():
            pattern = r'\b' + re.escape(term.replace(' ', '_').upper()) + r'\b'
            code_joined = re.sub(pattern, actual_col, code_joined, flags=re.IGNORECASE)

        # CORRECTION CRITIQUE: Remplacer les anciennes colonnes par les corrigées
        corrections = {
            'TEMPS_ATTENTE_ACCOSTAGE_HEURES': 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE',
            'DATE_RADE': 'DATETIME_RADE',
            'DATE_ACCOSTAGE': 'DATETIME_ACCOSTAGE',
            'TEMPS_ENTREE_ACCOSTAGE_HEURES': 'TEMPS_ENTREE_ACCOSTAGE_HEURES_CORRIGE',
            'TEMPS_SEJOUR_QUAI_HEURES': 'TEMPS_SEJOUR_QUAI_HEURES_CORRIGE'
        }
        for old_col, new_col in corrections.items():
            if old_col in code_joined:
                code_joined = code_joined.replace(f"'{old_col}'", f"'{new_col}'")
                code_joined = code_joined.replace(f'"{old_col}"', f'"{new_col}"')
                print(f"🔧 Correction automatique: {old_col} → {new_col}")

        # Intelligent code optimization based on question
        code_joined = self._optimize_code_for_question(code_joined, question)

        # Always ensure result assignment
        if not any(line.strip().startswith("result") for line in code_joined.split('\n')):
            if assign_lines and not code_joined.strip().endswith("result"):
                code_joined += f"\nresult = {assign_lines[-1].split('=')[0].strip()}"

        # FORÇAGE DU GRAPHIQUE : injecter le plot si demandé et absent
        if any(word in question_lower for word in ["graph", "graphique", "plot", "courbe", "visualiser"]):
            code_lines = code_joined.splitlines()
            has_plot = any("plt.show()" in l or ".plot(" in l for l in code_lines)
            plot_code = "\nimport matplotlib.pyplot as plt\nif isinstance(result, (pd.Series, pd.DataFrame)):\n    result.plot(kind='bar', figsize=(12,6))\n    plt.title(\"Graphique demandé par l'utilisateur\")\n    plt.tight_layout()\n    plt.show()"
            if not has_plot:
                code_joined = code_joined.rstrip() + plot_code

        return code_joined
    
    def _optimize_code_for_question(self, code: str, question: str) -> str:
        """Optimize code based on question context"""
        question_lower = question.lower()
        
        # Optimize temporal analysis
        if 'évolution' in question_lower or 'tendance' in question_lower:
            if 'DATETIME_RADE' in code and 'groupby' not in code:
                code = code.replace('DATETIME_RADE', 'DATETIME_RADE').replace(
                    'df[', 'df.groupby(df[\'DATETIME_RADE\'].dt.year).size() if \'DATETIME_RADE\' in df.columns else df['
                )
        
        # Optimize wait time calculations - FORCER l'utilisation des colonnes corrigées
        if any(word in question_lower for word in ['attente', 'délai', 'temps']):
            # Force l'utilisation de la colonne corrigée avec filtrage
            corrected_code = """
# Utilisation de la colonne temps d'attente corrigée avec filtrage
temps_data = df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
temps_data = temps_data[(temps_data >= 0) & (temps_data <= 240)]
result = temps_data.mean()
"""
            code = corrected_code
        
        # Fix common errors
        if 'nlargest' in code and 'value_counts' not in code:
            code = code.replace('.nlargest(', '.value_counts().head(')
        
        return code
    
    def _execute_with_advanced_retry(self, code: str, max_retries: int = 3) -> Tuple[Any, Optional[str]]:
        """Execute code with advanced retry mechanism"""
        local_vars = {"df": self.df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
        
        for attempt in range(max_retries):
            try:
                exec(code, {"pd": pd, "np": np, "plt": plt, "sns": sns}, local_vars)
                
                # Show plots if generated
                if plt.get_fignums():
                    plt.show()
                
                result = local_vars.get("result", "Code exécuté avec succès")
                return result, None
                
            except Exception as e:
                error_msg = str(e)
                self.error_history.append(error_msg)
                
                if attempt < max_retries - 1:
                    # Advanced error correction
                    corrected_code = self._attempt_error_correction(code, error_msg)
                    if corrected_code != code:
                        print(f"🔧 Correction automatique tentative {attempt + 1}")
                        code = corrected_code
                        continue
                
                return None, f"Erreur après {attempt + 1} tentatives: {error_msg}"
        
        return None, f"Échec après {max_retries} tentatives"
    
    def _attempt_error_correction(self, code: str, error_msg: str) -> str:
        """Attempt automatic error correction"""
        
        # KeyError correction
        if "KeyError" in error_msg:
            match = re.search(r"KeyError: '([^']+)'", error_msg)
            if match:
                wrong_col = match.group(1)
                closest_col = self._find_closest_column(wrong_col)
                if closest_col:
                    print(f"   🔧 Correction: '{wrong_col}' → '{closest_col}'")
                    code = code.replace(f"'{wrong_col}'", f"'{closest_col}'")
                    code = code.replace(f'"{wrong_col}"', f'"{closest_col}"')
        
        # Method error correction
        elif "Cannot use method 'nlargest' with dtype object" in error_msg:
            print("   🔧 Correction: nlargest → value_counts().head()")
            code = code.replace('.nlargest(', '.value_counts().head(')
        
        # Empty result correction
        elif "empty" in error_msg.lower():
            if ".mean()" in code:
                code = code.replace(".mean()", ".dropna().mean()")
            elif ".sum()" in code:
                code = code.replace(".sum()", ".dropna().sum()")
        
        return code
    
    def _find_closest_column(self, wrong_col: str) -> Optional[str]:
        """Find closest matching column name"""
        matches = get_close_matches(wrong_col, self.df.columns.tolist(), n=1, cutoff=0.6)
        return matches[0] if matches else None
    
    def _update_avg_response_time(self, execution_time: float):
        """Update average response time"""
        current_avg = self.performance_stats["avg_response_time"]
        total_queries = self.performance_stats["total_queries"]
        
        if total_queries == 1:
            self.performance_stats["avg_response_time"] = execution_time
        else:
            new_avg = ((current_avg * (total_queries - 1)) + execution_time) / total_queries
            self.performance_stats["avg_response_time"] = new_avg
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'data_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'calculated_columns': len([c for c in self.df.columns if c.startswith('TEMPS_') or c.startswith('DATETIME_')]),
                'date_range': {
                    'start': self.df['DATETIME_RADE'].min() if 'DATETIME_RADE' in self.df.columns else None,
                    'end': self.df['DATETIME_RADE'].max() if 'DATETIME_RADE' in self.df.columns else None
                }
            },
            'performance': self.performance_stats,
            'data_quality': {
                'error_count': len(self.error_history),
                'recent_errors': self.error_history[-3:] if self.error_history else [],
                'critical_columns_status': {
                    col: self.df[col].notna().sum() if col in self.df.columns else 'Missing'
                    for col in ['NOM_NAVIRE', 'TYPE_NAVIRE', 'DATETIME_RADE', 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE']
                }
            },
            'available_templates': list(self.analysis_templates.keys())
        }
    
    def interactive_session(self):
        """Enhanced interactive session"""
        print("=== 🚢 AGENT PORTUAIRE HYBRIDE CORRIGÉ - SESSION INTERACTIVE ===")
        print("Traitement corrigé des dates complexes + calculs de durées précis")
        print("\nCommandes spéciales:")
        print("- 'status' : État du système")
        print("- 'columns' : Liste des colonnes")
        print("- 'templates' : Analyses pré-configurées")
        print("- 'performance' : Statistiques de performance")
        print("- 'sample' : Échantillon des données")
        print("- 'comparison' : Comparaison ancien/nouveau calcul")
        print("- 'exit' : Quitter")
        
        print("\n💡 Exemples de questions intelligentes:")
        print("- Le temps d'attente moyen entre l'arrivée et l'accostage au quai")
        print("- Évolution mensuelle des escales")
        print("- Top 10 des navires par nombre d'escales")
        print("- Répartition par type de navire")
        print("- Statistiques générales du port")
        
        while True:
            question = input("\n🤖 Votre question: ").strip()
            
            if question.lower() == 'exit':
                self._show_session_summary()
                print("👋 Session terminée!")
                break
                
            elif question.lower() == 'status':
                status = self.get_system_status()
                print(json.dumps(status, indent=2, default=str))
                
            elif question.lower() == 'columns':
                self._show_columns_info()
                
            elif question.lower() == 'templates':
                self._show_available_templates()
                
            elif question.lower() == 'performance':
                self._show_performance_stats()
                
            elif question.lower() == 'sample':
                self._show_data_sample()
                
            elif question.lower() == 'comparison':
                self._show_calculation_comparison()
                
            else:
                # Process the actual question
                result = self.intelligent_query_processing(question)
                self._display_result(result)
    
    def _show_calculation_comparison(self):
        """Show comparison between old and corrected calculations"""
        print("\n🔍 COMPARAISON DES CALCULS:")
        
        if 'TEMPS_ATTENTE_ACCOSTAGE_HEURES' in self.df.columns and 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE' in self.df.columns:
            ancien = self.df['TEMPS_ATTENTE_ACCOSTAGE_HEURES'].dropna()
            nouveau = self.df['TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE'].dropna()
            nouveau_filtre = nouveau[(nouveau >= 0) & (nouveau <= 240)]
            
            print(f"Temps d'attente - Ancien calcul:")
            print(f"  - Nombre de valeurs: {len(ancien)}")
            print(f"  - Moyenne: {ancien.mean():.2f}h")
            print(f"  - Médiane: {ancien.median():.2f}h")
            print(f"  - Min/Max: {ancien.min():.2f}h / {ancien.max():.2f}h")
            
            print(f"\nTemps d'attente - Nouveau calcul (corrigé):")
            print(f"  - Nombre de valeurs: {len(nouveau)}")
            print(f"  - Moyenne: {nouveau.mean():.2f}h")
            print(f"  - Médiane: {nouveau.median():.2f}h")
            print(f"  - Min/Max: {nouveau.min():.2f}h / {nouveau.max():.2f}h")
            
            print(f"\nTemps d'attente - Nouveau calcul (filtré 0-240h):")
            print(f"  - Nombre de valeurs: {len(nouveau_filtre)}")
            print(f"  - Moyenne: {nouveau_filtre.mean():.2f}h")
            print(f"  - Médiane: {nouveau_filtre.median():.2f}h")
            print(f"  - Min/Max: {nouveau_filtre.min():.2f}h / {nouveau_filtre.max():.2f}h")
            
            print(f"\n📊 Différence moyenne: {abs(ancien.mean() - nouveau_filtre.mean()):.2f}h")
        else:
            print("❌ Colonnes de comparaison non disponibles")
    
    def _show_session_summary(self):
        """Show session summary"""
        stats = self.performance_stats
        success_rate = (stats["successful_queries"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
        
        print(f"\n📊 RÉSUMÉ DE SESSION:")
        print(f"- Requêtes totales: {stats['total_queries']}")
        print(f"- Requêtes réussies: {stats['successful_queries']}")
        print(f"- Taux de succès: {success_rate:.1f}%")
        print(f"- Temps moyen de réponse: {stats['avg_response_time']:.2f}s")
        
        if self.error_history:
            print(f"- Erreurs rencontrées: {len(self.error_history)}")
    
    def _show_columns_info(self):
        """Show comprehensive columns information"""
        print("\n📋 INFORMATIONS SUR LES COLONNES:")
        
        original_cols = []
        calculated_cols = []
        datetime_cols = []
        
        for col in self.df.columns:
            if col.startswith('DATETIME_'):
                datetime_cols.append(col)
            elif col.startswith('TEMPS_') and 'CORRIGE' in col:
                calculated_cols.append(col)
            else:
                original_cols.append(col)
        
        print(f"\n🗂️  Colonnes originales ({len(original_cols)}):")
        for i, col in enumerate(original_cols[:15], 1):
            valid_count = self.df[col].notna().sum()
            print(f"  {i:2d}. {col} ({valid_count}/{len(self.df)} valides)")
        
        if len(original_cols) > 15:
            print(f"  ... et {len(original_cols) - 15} autres colonnes")
        
        if datetime_cols:
            print(f"\n🕕 Colonnes DateTime complètes ({len(datetime_cols)}):")
            for i, col in enumerate(datetime_cols, 1):
                valid_count = self.df[col].notna().sum()
                print(f"  {i:2d}. {col} ({valid_count}/{len(self.df)} valides)")
        
        if calculated_cols:
            print(f"\n🧮 Colonnes calculées CORRIGÉES ({len(calculated_cols)}):")
            for i, col in enumerate(calculated_cols, 1):
                valid_count = self.df[col].notna().sum()
                print(f"  {i:2d}. {col} ({valid_count}/{len(self.df)} valides)")
    
    def _show_available_templates(self):
        """Show available analysis templates"""
        print("\n📋 ANALYSES PRÉ-CONFIGURÉES:")
        
        for i, (name, info) in enumerate(self.analysis_templates.items(), 1):
            validation_status = "✅" if info['validation'](self.df) else "❌"
            keywords = ", ".join(info['keywords'][:3])
            print(f"  {i}. {name} {validation_status}")
            print(f"     Mots-clés: {keywords}")
            print(f"     Visualisation: {'Oui' if info.get('visualization', False) else 'Non'}")
    
    def _show_performance_stats(self):
        """Show detailed performance statistics"""
        stats = self.performance_stats
        print(f"\n📈 STATISTIQUES DE PERFORMANCE:")
        print(f"- Requêtes traitées: {stats['total_queries']}")
        print(f"- Succès: {stats['successful_queries']}")
        print(f"- Échecs: {stats['total_queries'] - stats['successful_queries']}")
        
        if stats['total_queries'] > 0:
            success_rate = stats['successful_queries'] / stats['total_queries'] * 100
            print(f"- Taux de succès: {success_rate:.1f}%")
            print(f"- Temps moyen: {stats['avg_response_time']:.2f}s")
        
        if self.error_history:
            print(f"\n⚠️  Erreurs récentes:")
            for error in self.error_history[-5:]:
                print(f"  - {error}")
    
    def _show_data_sample(self):
        """Show data sample with key information"""
        print(f"\n🔍 ÉCHANTILLON DES DONNÉES:")
        
        # Key columns to show
        key_cols = ['NOM_NAVIRE', 'TYPE_NAVIRE', 'DATETIME_RADE', 'PAVILLON', 'TEMPS_ATTENTE_ACCOSTAGE_HEURES_CORRIGE']
        available_cols = [col for col in key_cols if col in self.df.columns]
        
        if available_cols:
            sample_df = self.df[available_cols].head(5)
            print(sample_df.to_string(index=False))
        else:
            print("Colonnes clés non disponibles")
        
        print(f"\nTotal: {len(self.df)} lignes")
        if 'DATETIME_RADE' in self.df.columns:
            date_range = f"{self.df['DATETIME_RADE'].min()} à {self.df['DATETIME_RADE'].max()}"
            print(f"Période: {date_range}")
    
    def _display_result(self, result: Dict[str, Any]):
        """Display analysis result with enhanced formatting and clarity"""
        if result['success']:
            print(f"✅ Analyse réussie ({result['method']})")
            if 'template_used' in result:
                print(f"📋 Template utilisé : {result['template_used']}")
            res = result.get('result', None)
            if res is not None:
                print("\n📊 RÉSULTAT :")
                if isinstance(res, pd.Series):
                    print(res.to_string())
                elif isinstance(res, pd.DataFrame):
                    print(res.head(20).to_string(index=False))
                elif isinstance(res, dict):
                    for k, v in res.items():
                        print(f"- {k}: {v}")
                else:
                    print(res)
            else:
                print("Aucun résultat à afficher.")
        else:
            print(f"❌ Échec de l'analyse ({result.get('method', 'unknown')})")
            print(f"Erreur: {result['error']}")
            if 'template_failed' in result.get('method', ''):
                print("Essayez une autre question ou vérifiez les colonnes disponibles.")
            elif 'llm_failed' in result.get('method', ''):
                print("Le code généré n'a pas pu être exécuté. Essayez de reformuler la question.")


# Usage and initialization
if __name__ == "__main__":
    print("🚀 Initialisation de l'Agent Portuaire Hybride Corrigé...")
    try:
        agent = HybridPortAgent("anp_dataset_clean_sample.csv")
    except FileNotFoundError:
        print("❌ Fichier CSV introuvable. Vérifiez le chemin.")
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        print("Vérifiez que toutes les dépendances sont installées et que le fichier CSV est accessible.")
    else:
        print("\nVous pouvez maintenant poser vos questions à l'agent. Tapez 'exit' pour quitter.")
        agent.interactive_session()