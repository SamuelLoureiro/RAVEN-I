# Planification — *Learning Symbolic Representation with RAVEN-I*

> Module IAS (ENIB) · Binôme · **3 semaines** : du 16/11 au 06/12/2026

## 1. Objectif

À partir des **seules images** d'une matrice RAVEN-I (3×3, dernière case manquante, 8 candidats) :

1. **extraire les attributs** de chaque case : *Number, Position, Type, Size, Color* ;
2. **inférer les règles** sur chaque attribut : *Constant, Progression, Arithmetic, Distribute_Three* ;
3. **choisir la bonne réponse** (0–7).

Supervision : les fichiers `.xml` (attributs et règles vérité terrain). Split imposé : **60 / 20 / 20**.

## 2. Organisation

**Le binôme travaille ensemble sur la même approche**, en pair programming et en échangeant régulièrement les rôles (conception, implémentation, tests, analyse), afin que les deux membres maîtrisent l'ensemble du projet.

* **Architecture commune (pipeline en 2 étapes) :**
  * **Étape 1 — Perception :** extraction des attributs (*Number, Position, Type, Size, Color*) à partir des images (CNN multi-têtes / ResNet-18).
  * **Étape 2 — Raisonnement :** inférence des règles (*Constant, Progression, Arithmetic, Distribute_Three*) et sélection de la réponse (0–7).
* **Comparaison de modèles de raisonnement** (demandée par le sujet) :
  * baseline par **solveur symbolique** (pour valider la compréhension des règles sur la vérité terrain) ;
  * évaluation et comparaison **arbre de décision vs Transformer** sur les attributs (vérité terrain puis prédits).
* **Suivi commun :**
  * même dataset (`--seed 42`) et **même split 60 / 20 / 20** ;
  * mêmes métriques : accuracy des attributs, accuracy des règles, accuracy de résolution, F1-score, matrices de confusion ;
  * commits réguliers des deux membres du binôme sur le dépôt partagé.

## 3. Planning

### Semaine 1 (16/11 → 22/11) : données et compréhension
- Environnement (`requirements.txt`), génération de `test-1k` puis `RAVEN-10k` (avec `--seed 42`).
- Notebook `exploration.ipynb` : grille 3×3 + 8 candidats, retrouver la réponse à la main.
- Décoder `meta_matrix`, `meta_target`, `meta_answer_mods` et le XML (règles par **lignes uniquement**).
- Statistiques : distribution des règles, configurations, types d'objets, vérification du split 60/20/20.
- **Fin de semaine** : un raisonnement qui marche à partir des **attributs vérité terrain** (solveur symbolique proche de 100 % pour valider les règles, puis premier prototype d'arbre de décision / Transformer).

### Semaine 2 (23/11 → 29/11) : perception sur `center_single`
- Modèle de perception sur `center_single` : d'abord **un seul attribut** (`Type`), puis tous les attributs.
- Brancher la perception sur le raisonnement → pipeline complet image → réponse.
- Comparer le raisonnement sur attributs **vérité terrain** vs **prédits** (solveur symbolique, arbre de décision, Transformer).
- **Fin de semaine** : pipeline complet fonctionnel sur `center_single` + premières métriques.

### Semaine 3 (30/11 → 06/12) : généralisation, comparaison, rendu
- **Lun–Mar** : étendre aux autres configurations (2 composants, puis `distribute_four/nine` et configurations imbriquées).
- **Mer** : évaluation finale sur le test set. **Arrêt des nouvelles fonctionnalités.**
- **Jeu** : tableau comparatif des modèles testés, analyse d'erreurs, README.
- **Ven** : slides, démo, répétition.
- **Week-end** : corrections finales.

## 4. Comparaison finale (à remplir)

| Accuracy (test) | Solveur symbolique | Arbre de décision | Transformer |
|---|---|---|---|
| Attributs (moyenne — étape 1) | — | — | — |
| Règles, attributs vérité terrain | | | |
| Règles, attributs prédits | | | |
| Résolution (hasard = 12,5 %) | | | |

Questions d'analyse (demandées par le sujet) :
- Quelles **règles** sont les plus difficiles ? Quels **attributs** ? Quelles **configurations** ?
- Comment les erreurs de perception se propagent-elles dans le raisonnement ?
- Arbre de décision vs Transformer vs solveur symbolique : font-ils les mêmes erreurs ? Interprétabilité vs performance ?

## 5. Rendu

- **Code** : génération, notebook d'exploration, modèles, entraînement, évaluation.
- **README (≤ 2 pages, style liste)** : noms, implémentation, variantes, « avec plus de temps… et comment », installation et exécution reproductibles, bibliothèques utilisées, **déclaration d'utilisation d'IA**.
- **Soutenance** : approche retenue, résultats comparatifs (tableaux, courbes), analyse d'erreurs, limites, éthique (par ex. ce que signifie qu'une IA « réussit un test de QI », raccourcis dans les datasets), **démo** du pipeline.

> [!IMPORTANT]
> - Commits **réguliers** sur au moins 2 semaines : c'est un critère de notation.
> - Chacun doit pouvoir **expliquer chaque ligne de son code**. Sans ça, pas de note.
