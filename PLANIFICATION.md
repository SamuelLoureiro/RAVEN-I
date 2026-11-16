# Planification — *Learning Symbolic Representation with RAVEN-I*

> Module IAS (ENIB) · Binôme · **3 semaines** : du 16/11 au 06/12/2026

## 1. Objectif

À partir des **seules images** d'une matrice RAVEN-I (3×3, dernière case manquante, 8 candidats) :

1. **extraire les attributs** de chaque case : *Number, Position, Type, Size, Color* ;
2. **inférer les règles** sur chaque attribut : *Constant, Progression, Arithmetic, Distribute_Three* ;
3. **choisir la bonne réponse** (0–7).

Supervision : les fichiers `.xml` (attributs et règles vérité terrain). Split imposé : **60 / 20 / 20**.

## 2. Organisation

**Chacun réalise le projet en entier, seul, sur sa propre branche**, avec une approche différente. On compare les deux à la fin.

| | Personne A | Personne B |
|---|---|---|
| Branche | `A` | `B` |
| Perception (étape 1) | CNN léger multi-têtes (une tête par attribut) | ResNet-18 multi-têtes |
| Raisonnement (étape 2) | Solveur symbolique + arbre de décision | Petit Transformer sur les attributs |
| Esprit | Interprétable | Apprentissage profond |

Le sujet demande justement d'« évaluer transformer ou arbres de décision » : la comparaison A vs B y répond directement.

**Les seuls points à fixer ensemble (lundi 16/11), pour pouvoir comparer :**
- même dataset (même seed, même nombre d'échantillons) et **même split** ;
- mêmes métriques : accuracy des attributs, accuracy des règles, accuracy de résolution, F1, matrices de confusion ;
- un **point de 15 min par semaine** pour partager les avancées et les difficultés.

## 3. Planning (identique pour A et B)

### Semaine 1 (16/11 → 22/11) : données et compréhension
- Environnement (Python 3.9, `requirements.txt`), génération de `test-1k` puis `RAVEN-10k`.
- Notebook `exploration.ipynb` : grille 3×3 + 8 candidats, retrouver la réponse à la main.
- Décoder `meta_matrix`, `meta_target`, `meta_answer_mods` et le XML.
- Statistiques : distribution des règles, configurations, types d'objets.
- Chargement des données + labels issus du XML, split 60/20/20.
- **Fin de semaine** : un raisonnement qui marche à partir des **attributs vérité terrain** (A : solveur symbolique ; B : premier transformer). Proche de 100 % pour A, sinon on a mal compris les règles.

### Semaine 2 (23/11 → 29/11) : perception sur `center_single`
- Modèle de perception sur `center_single` : d'abord **un seul attribut** (`Type`), puis tous.
- Brancher la perception sur le raisonnement → pipeline complet image → réponse.
- Comparer : raisonnement sur attributs **vérité terrain** vs **prédits**.
- A : ajouter l'arbre de décision. B : entraîner le transformer sur attributs prédits.
- **Fin de semaine** : pipeline complet fonctionnel sur `center_single` + premières métriques.

### Semaine 3 (30/11 → 06/12) : généralisation, comparaison, rendu
- **Lun–Mar** : étendre aux autres configurations (2 composants, puis `distribute_four/nine`).
- **Mer** : évaluation finale sur le test set. **Arrêt des nouvelles fonctionnalités.**
- **Jeu** (ensemble) : tableau comparatif A vs B, analyse d'erreurs, README.
- **Ven** (ensemble) : slides, démo, répétition.
- **Week-end** : corrections finales.

## 4. Comparaison finale (à remplir)

| Accuracy (test) | A | B |
|---|---|---|
| Attributs (moyenne) | | |
| Règles, attributs vérité terrain | | |
| Règles, attributs prédits | | |
| Résolution (hasard = 12,5 %) | | |

Questions d'analyse (demandées par le sujet) :
- Quelles **règles** sont les plus difficiles ? Quels **attributs** ? Quelles **configurations** ?
- Comment les erreurs de perception se propagent-elles dans le raisonnement ?
- A et B font-ils les mêmes erreurs ? Interprétabilité vs performance ?

## 5. Rendu

- **Code** : génération, notebook d'exploration, modèles, entraînement, évaluation.
- **README (≤ 2 pages, style liste)** : noms, implémentation, variantes, « avec plus de temps… et comment », installation et exécution reproductibles, bibliothèques utilisées, **déclaration d'utilisation d'IA**.
- **Soutenance** : approches, résultats (tableaux, courbes), analyse d'erreurs, limites, éthique (par ex. ce que signifie qu'une IA « réussit un test de QI », raccourcis dans les datasets), **démo** des deux pipelines.

> [!IMPORTANT]
> - Commits **réguliers** sur au moins 2 semaines : c'est un critère de notation.
> - Chacun doit pouvoir **expliquer chaque ligne de son code**. Sans ça, pas de note.
