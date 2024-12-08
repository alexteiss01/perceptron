# Import packages
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import Perceptron

# Importing the dataset
data = pd.read_csv('bcw_data.csv')

# Removing the last column as it is empty
data = data.drop('Unnamed: 32', axis=1)

# Checking for null values
data.isnull().sum()

# Removing the 'id' column as it is unique
data = data.drop('id', axis=1)

# Encoding the 'diagnosis' column as numerical values
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Checking the different values contained in the diagnosis column
print(data['diagnosis'].value_counts())

# Data Visualization
sns.countplot(x='diagnosis', data=data)
plt.title("Distribution of the Target Variable (Diagnosis)")
plt.show()

# Interpretation of diagnosis distribution
# Le déséquilibre entre les classes bénignes (`B`) et malignes (`M`) peut biaiser le modèle en faveur de la classe majoritaire.
# Une solution possible est d'appliquer un sous-échantillonnage ou un sur-échantillonnage, ou d'utiliser des métriques adaptées comme le F1-score.

# Histogram analysis for 'mean', 'se', and 'worst' features
mean_features = [col for col in data.columns if '_mean' in col]
se_features = [col for col in data.columns if '_se' in col]
worst_features = [col for col in data.columns if '_worst' in col]

# Plot histograms for mean features
data[mean_features].hist(bins=15, figsize=(15, 12))
plt.suptitle("Distribution des caractéristiques - Moyennes (Mean)")
plt.show()

# Interpretation of mean features
# Les distributions comme `radius_mean`, `perimeter_mean`, et `area_mean` montrent des asymétries marquées avec des valeurs plus élevées pour les tumeurs malignes.
# Les variables comme `compactness_mean` et `concave points_mean` présentent des pics prononcés pour les tumeurs bénignes, mais des queues longues vers des valeurs plus élevées, suggérant une différenciation entre les diagnostics.
# Conclusion : Les tumeurs malignes tendent à être plus grandes et à avoir des contours plus irréguliers.

# Boxplot analysis
plt.figure(figsize=(15, 20))
for i, feature in enumerate(mean_features, 1):
    plt.subplot(len(mean_features) // 3 + 1, 3, i)
    sns.boxplot(x='diagnosis', y=feature, data=data, palette='coolwarm')
    plt.title(f"Boxplot of {feature} by Diagnosis")

plt.tight_layout()
plt.show()

# Interpretation of boxplots
# Les boxplots montrent une séparation claire entre les diagnostics bénins (`B`) et malins (`M`) pour plusieurs caractéristiques comme `radius_mean`, `perimeter_mean`, `concave points_mean`, etc.
# Des valeurs extrêmes (outliers) sont présentes, particulièrement dans les tumeurs malignes, ce qui reflète la nature hétérogène de ces tumeurs.
# Conclusion : Les caractéristiques comme `radius_mean` et `concave points_mean` montrent une différence significative entre les classes, ce qui en fait de bons candidats pour la modélisation.

# Scatterplot analysis
feature_pairs = list(itertools.combinations(mean_features, 2))

plt.figure(figsize=(5 * 3, 5 * len(feature_pairs) // 3))
for i, (feature_x, feature_y) in enumerate(feature_pairs[:6], 1):  # Limit to the first 6 pairs
    plt.subplot(2, 3, i)
    sns.scatterplot(x=feature_x, y=feature_y, hue='diagnosis', data=data, palette='coolwarm', s=20)
    plt.title(f"{feature_x} vs {feature_y}")

plt.tight_layout()
plt.show()

# Interpretation of scatterplots
# Certaines variables, comme `radius_mean` et `perimeter_mean`, présentent des relations linéaires claires, confirmant une forte corrélation.
# Les clusters dans les scatterplots montrent une différenciation partielle entre les diagnostics, mais avec une superposition notable.
# Conclusion : Ces relations suggèrent que certaines variables capturent des informations redondantes, ce qui justifie l'application de l'ACP.

# Analysis of worst features
worst_features = [col for col in data.columns if '_worst' in col]

# Histogram of worst features
data[worst_features].hist(bins=15, figsize=(15, 12), color='green', alpha=0.7)
plt.suptitle("Distribution des caractéristiques - Valeurs Maximales (Worst)")
plt.show()

# Interprétation des histogrammes des caractéristiques `worst`
# - Les caractéristiques liées à la taille (`radius_worst`, `perimeter_worst`, `area_worst`) montrent des asymétries avec une concentration de valeurs faibles pour la majorité des observations.
#   Les valeurs plus élevées correspondent généralement aux tumeurs malignes.
# - Les caractéristiques liées à la régularité (`smoothness_worst`, `symmetry_worst`) présentent des distributions plus symétriques.
# - Les caractéristiques `concavity_worst` et `concave points_worst` ont des queues longues, typiques des contours irréguliers associés aux tumeurs malignes.
# Conclusion : Les caractéristiques `worst` permettent d’identifier des tumeurs malignes grâce à leurs valeurs extrêmes, en particulier pour les variables liées à la taille et aux contours.


# Boxplots for worst features by diagnosis
plt.figure(figsize=(15, 20))
for i, feature in enumerate(worst_features, 1):
    plt.subplot(len(worst_features) // 3 + 1, 3, i)
    sns.boxplot(x='diagnosis', y=feature, data=data, palette='coolwarm')
    plt.title(f"Boxplot of {feature} by Diagnosis")

plt.tight_layout()
plt.show()

# Interprétation des boxplots des caractéristiques `worst` par diagnostic
# - Les valeurs de `radius_worst`, `perimeter_worst`, et `area_worst` sont significativement plus élevées pour les tumeurs malignes (`diagnosis = 1`).
# - Les variables `concavity_worst` et `concave points_worst` montrent une distinction nette entre les deux classes. Les tumeurs malignes ont des valeurs beaucoup plus élevées, reflétant des contours irréguliers et des indentations prononcées.
# - Les variables liées à la régularité (`smoothness_worst`, `symmetry_worst`) montrent moins de variabilité entre les diagnostics, mais les valeurs maximales sont légèrement plus élevées pour les tumeurs malignes.
# Conclusion : Les caractéristiques `worst` capturent bien les différences entre les classes, notamment en ce qui concerne la taille et l’irrégularité des contours.


# Correlation heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(data=data.corr(), annot=True, fmt='.1f', cmap='Greens')
plt.title("Heatmap of Feature Correlations")
plt.show()

# Interpretation of correlation matrix
# Les caractéristiques liées à la taille (`radius_mean`, `perimeter_mean`, `area_mean`) montrent de fortes corrélations entre elles.
# Les caractéristiques des contours (`compactness_mean`, `concavity_mean`, `concave points_mean`) sont également fortement corrélées.
# Conclusion : Ces groupes de variables redondantes peuvent être efficacement capturés par l'ACP, réduisant la dimensionnalité sans perdre d'information importante.

# Standardisation des données
X = data.drop(columns=['diagnosis'])  # Exclure la colonne cible
y = data['diagnosis']  # La cible est la variable 'diagnosis'

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Application de l'ACP
pca = PCA(n_components=4)  # 4 composantes principales
X_pca = pca.fit_transform(X_scaled)

# Variance expliquée par chaque composante principale
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Visualisation de la variance expliquée
plt.figure(figsize=(10, 6))
sns.barplot(x=['PC1', 'PC2', 'PC3', 'PC4'], y=explained_variance * 100, palette='Blues_d')
plt.plot(['PC1', 'PC2', 'PC3', 'PC4'], cumulative_variance * 100, color="red", marker="o", label="Cumulative Variance")
plt.ylabel("Variance expliquée (%)")
plt.xlabel("Composantes principales")
plt.ylim(0, 100)
plt.title("Variance expliquée par les 4 premières composantes principales")
plt.legend()
plt.show()

# Contribution des variables aux composantes principales
loadings = pd.DataFrame(pca.components_, columns=X.columns, index=['PC1', 'PC2', 'PC3', 'PC4'])

# Affichage des contributions des variables pour PC1 et PC2
idex=[]
for pc in ['PC1', 'PC2', 'PC3', 'PC4']:
    top_contributors = loadings.loc[pc].abs().sort_values(ascending=False)[:5]
    idex.append(top_contributors.index[0])
    idex.append(top_contributors.index[1])
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_contributors.index, y=top_contributors.values * 100, palette="Blues_d")
    plt.title(f"Top 5 contributions pour {pc}")
    plt.xlabel("Variables")
    plt.ylabel("Contribution (%)")
    plt.xticks(rotation=45)
    plt.show()
    
    # Interpretation of the 4 principal components
# Interprétation des 4 composantes principales :
# 
# 1. PC1 (Première Composante Principale) :
#    - Variance expliquée : Environ 21.8 %.
#    - Variables principales : concave points_mean, perimeter_mean, area_mean, concavity_mean, compactness_mean.
#    - Interprétation : PC1 capture les aspects liés à la taille et à la régularité des contours des tumeurs, avec une forte contribution des tumeurs malignes.
#
# 2. PC2 (Deuxième Composante Principale) :
#    - Variance expliquée : Environ 18.9 %.
#    - Variables principales : radius_mean, texture_mean, smoothness_mean, symmetry_mean, fractal_dimension_mean.
#    - Interprétation : PC2 capture les caractéristiques de texture et de complexité des contours.
#
# 3. PC3 (Troisième Composante Principale) :
#    - Variance expliquée : Environ 16.3 %.
#    - Variables principales : radius_se, texture_se, perimeter_se, smoothness_se, concavity_se.
#    - Interprétation : PC3 met en avant la variabilité locale des mesures.
#
# 4. PC4 (Quatrième Composante Principale) :
#    - Variance expliquée : Environ 13.0 %.
#    - Variables principales : fractal_dimension_worst, concave points_worst, symmetry_worst, compactness_worst, radius_worst.
#    - Interprétation : PC4 capture les valeurs extrêmes associées aux tumeurs malignes.

#on prend les deux meilleurs contributeur de chaque catégorie

print(idex)
toevaluate=[]
for i in idex :
    toevaluate.append(list(data[i]))
ToEvaluate=[]
for i in range(len(toevaluate[0])) :
    inp=[]
    for j in range(len(toevaluate)) :
        inp.append(toevaluate[j][i])
    ToEvaluate.append(inp)
p=Perceptron.perceptron(0.001,1,[1,1,1,1,1,1,1,1],Perceptron.sigmoide)
p.learn(ToEvaluate,list(y),4000,10)
print(p.moyevaluate(ToEvaluate,list(y)))