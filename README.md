# Détection d'Anomalies sur Graphe Attribué (Réseau Aérien)

Ce projet implémente et compare des approches d'apprentissage géométrique profond (Graph Neural Networks) pour la détection d'anomalies sur un graphe attribué représentant les aéroports mondiaux, leurs liaisons de vol et des attributs associés (population, coordonnées).

---

## Modèles implémentés

DOMINANT (Deep Anomaly Detection on Attributed Networks) :
   - Encodeur GCN multi-couches générant un embedding.
   - Décodeur de structure calculant la probabilité d'arêtes.
   - Décodeur d'attributs (MLP) reconstruisant les caractéristiques des noeuds.
   - Score d'anomalie hybride combinant l'erreur de reconstruction structurelle et l'erreur d'attributs.

GAE (*Graph Auto-Encoder*) :
   - Encodeur basé sur deux couches `GCNConv`.
   - Reconstruction de la matrice d'adjacence via produit scalaire et fonction de perte binaire (`recon_loss`).
   - Score d'anomalie basé sur l'erreur locale de prédiction des arêtes réelles.

---

## Protocole de Validation Crucial (Injection et Re-test)

Le pipeline contient une procédure à suivre qui permet d'injecter des anomalies synthétique pour valider le pouvoir du modele.

### 1. Première phase : Détection initiale (Graphe d'origine)
- Le modèle DOMINANT tourne sur `airportsAndCoordAndPop.graphml`.
- Les noeuds ayant un score d'anomalie élevé (score > 15 000, incluant les 10 anomalies majeures initiales telles que Lagos, Shanghai, etc.) sont identifiés et supprimés.
- Un graphe nettoyé intermédiaire (`nouveau_graph.graphml`) est alors sauvegardé.

### 2. Seconde phase : Injection d'anomalies extrêmes
- Dans la cellule précédant le GAE, on ajoute manuellement des noeuds aberrants (ex. aéroports fictifs avec plus de 1000 à 2000 aéroports connectés et des populations trop faible ou trop forte).
- Ce graphe enrichi est exporté sous :
  ```
  nouveau_graph_anomalies.graphml
  ```

### 3. Étape Obligatoire de Validation
Une fois le fichier `nouveau_graph_anomalies.graphml` généré :
1. Remontez au tout début du notebook (cellule de chargement NetworkX).
2. Remplacez le nom du fichier lu :
```python
   # Remplacer :
   G = nx.read_graphml("airportsAndCoordAndPop.graphml")
   # Par :
   G = nx.read_graphml("nouveau_graph_anomalies.graphml")
```
3. Ré-exécutez l'ensemble du pipeline.
4. Objectif : Vérifier que les modèles (DOMINANT puis GAE) font immédiatement remonter les nouveaux nœuds injectés en Top 10 des anomalies avec un score significativement détaché du reste du graphe (ex. score 23 000 contre < 18 000 pour les noeuds standards).

---

## Dépendances requises

Installez les bibliothèques nécessaires avec `pip` ou `conda` :

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install networkx matplotlib seaborn scikit-learn numpy pandas
```

---

## Fichiers du projet

- `airportsAndCoordAndPop.graphml` : Graphe brut d'origine.
- `nouveau_graph.graphml` : Graphe après suppression des anomalies naturelles (score > 15).
- `nouveau_graph_anomalies.graphml` : Graphe de test intégrant les anomalies structurelles et d'attributs injectées manuellement.
