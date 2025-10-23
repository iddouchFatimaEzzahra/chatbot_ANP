# 🚢 Agent Conversationnel Intelligent pour l'Optimisation des Opérations Portuaires

## 📌 Contexte du Projet
Ce projet a été réalisé dans le cadre d'un stage à l'**Agence Nationale des Ports (ANP)** du Maroc. Il vise à développer un agent conversationnel intelligent pour faciliter l'accès et l'analyse des données portuaires complexes.

## 🎯 Objectifs
- **Automatiser** l'analyse des données portuaires (navires, tonnages, temps d'attente, etc.)
- **Démocratiser** l'accès aux données via une interface conversationnelle naturelle
- **Combiner** analyse prédéfinie et intelligence artificielle générative
- **Visualiser** les indicateurs clés via un dashboard interactif

## 🏗️ Architecture du Système
### Approche Hybride
- **Templates analytiques** pour les requêtes récurrentes
- **LLM (Llama 3.1)** pour les questions complexes et non anticipées
- **Validation multi-niveaux** pour garantir la fiabilité

### Composants Principaux
1. **Module Python (`HybridPortAgent`)**
   - Nettoyage et préparation des données
   - Gestion des dates/heures complexes
   - Calcul des indicateurs de performance

2. **Interface Streamlit**
   - Dashboard interactif avec KPI
   - Visualisations dynamiques (Plotly)
   - Chatbot intégré en langage naturel

## 📊 Fonctionnalités
### 🎮 Interface Utilisateur
- Import de datasets CSV portuaires
- Tableau de bord avec indicateurs clés :
  - Nombre d'escales et navires uniques
  - Tonnages import/export
  - Temps d'attente moyen
- Visualisations temporelles et statistiques

### 💬 Chatbot Intelligent
- Réponses en langage naturel
- Génération automatique de code Python
- Correction d'erreurs intégrée
- Historique des sessions utilisateur

## 🛠️ Technologies Utilisées
- **Backend** : Python, Pandas, NumPy
- **IA** : Llama 3.1 (via Ollama)
- **Frontend** : Streamlit, Plotly
- **Visualisation** : PlantUML pour les diagrammes

## 📈 Résultats Obtenus
- **8 000+ escales** traitées couvrant 2020-2023
- **89% de taux de succès** global des analyses
- **Temps de réponse** :
  - Templates : 0.8s en moyenne
  - LLM : 2.3-4.2s selon la complexité
- **6 templates analytiques** prédéfinis

## 🚀 Défis Relevés
### Techniques
- **Gestion des dates multiples** avec séparateurs "|"
- **Fiabilité de la génération LLM** avec système de retry
- **Optimisation des performances** avec cache multi-niveaux

### Métier
- **Terminologie portuaire spécialisée** (rade, accostage, etc.)
- **Processus opérationnels complexes**
- **Validation métier des résultats**


## 🔮 Perspectives
- Intégration de modèles prédictifs
- Enrichissement automatique des templates
- Interopérabilité avec les systèmes existants de l'ANP
- Extension à d'autres ports marocains

---

*Projet réalisé par **Fatima-Ezzahra IDDOUCH** dans le cadre d'un stage à l'Agence Nationale des Ports (2024-2025)*
