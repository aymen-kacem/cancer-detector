import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score


# Charger les données du cancer du sein

cancer = load_breast_cancer()
X = cancer.data
Y = cancer.target
# Récupérer les noms des caractéristiques et la cible
features = cancer.feature_names
print("features names",features)
target = cancer.target_names
print("les classes sont",target)
target_dimension=cancer.target.shape
print("le dimension de target est",target_dimension)
data_dimension=cancer.data.shape
print("le dimension de data est",data_dimension)
Description=cancer.DESCR
print("description",Description)
# Créer un DataFrame à partir des données
cancer_df = pd.DataFrame(data=cancer.data, columns=features)

# Sélectionner un sous-ensemble de caractéristiques pour la matrice de corrélation
caracteristiques_selectionnees = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity']

# Créer un DataFrame avec uniquement les caractéristiques sélectionnées
df_caracteristiques_selectionnees = cancer_df[caracteristiques_selectionnees]

# Calculer la matrice de corrélation pour les caractéristiques sélectionnées
matrice_correlation_selectionnee = df_caracteristiques_selectionnees.corr()

# Tracer la heatmap de la matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(matrice_correlation_selectionnee, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation")
plt.show()#
mean_texture_index = np.where(features == "mean texture")[0][0]

# Plotting the histogram
plt.figure(figsize=(8,6))
plt.hist(X[:, mean_texture_index], bins=30, edgecolor='k')
plt.xlabel("mean texture")
plt.ylabel("frequency")
plt.title("Histogram")
plt.show()
#
plt.figure(figsize=(8, 6))
sns.kdeplot(X[:,mean_texture_index] , fill=True)
plt.xlabel("mean texture")
plt.ylabel("distribution de probabilité")
plt.title("distribution de probabilité de mean texture")
plt.show()


df_num = cancer_df[['mean area', 'worst area', 'area error']]

df_num.head()
for col in df_num.columns:
    sns.histplot(cancer_df[col],  kde=True, bins=30, palette='muted')
    plt.title(col)
    plt.show()

# Vérifier les outliers en utilisant le Z-score
z_scores = stats.zscore(cancer_df)
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)  # Utilisation du Z-score avec un seuil de 3
outliers = cancer_df[~filtered_entries]
print("Les valeurs aberant")
print(outliers)

# Vérifier la présence de valeurs nulles
has_null_values = cancer_df.isnull().any().any()
print("Le jeu de données contient des valeurs nulles (NaN): ", has_null_values)

# Supprimer les lignes avec des valeurs nulles si nécessaire
cancer_df.dropna(inplace=True)
##########################################################################
##########################################################################
# division du data
X_train, X_test, y_train, y_test = train_test_split(cancer_df, cancer.target, test_size=0.25, random_state=42)
print("Taille de l'ensemble d'apprentissage (X_train, y_train) :", X_train.shape, y_train.shape)
print("Taille de l'ensemble de test (X_test, y_test):", X_test.shape, y_test.shape)
#SVM
from sklearn.svm import SVC
from sklearn.metrics import classification_report

kernels = ['poly', 'linear', 'rbf']

for kernel in kernels:
    svm_model = SVC(kernel=kernel)
    svm_model.fit(X_train, y_train)  # Entraînement du modèle avec les données d'entraînement
    scores = cross_val_score(svm_model, X, Y, cv=4)
    print(f"Score de validation croisée pour le noyau {kernel}: {np.mean(scores)}")

    # Ajouter le code pour la prédiction avec SVM après l'entraînement
    y_pred_svm = svm_model.predict(X_test)
    print("classification svm", classification_report(y_test, y_pred_svm))
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    print("Matrice de confusion pour SVM:\n", conf_matrix_svm)

#  Déterminer le noyau optimal
# Définissez la grille des hyperparamètres à explorer
param_grid = {'kernel': kernels}
# Utilisez GridSearchCV pour trouver le meilleur noyau
grid_search = GridSearchCV(svm_model, param_grid, cv=4)
grid_search.fit(X, Y)
# Affichez le meilleur noyau et le meilleur score
print("Meilleur noyau pour SVM basé sur la validation croisée :", grid_search.best_params_['kernel'])
print("Meilleur score de validation croisée :", grid_search.best_score_)


# Entraîner un modèle KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_model.fit(X_train, y_train)
scores1 = cross_val_score(knn_model, X, Y, cv=4)
print("validation croisee de modele knn",scores1)
param_grid1 = {
    'n_neighbors': [3, 5, 7],
    'metric': ['euclidean', 'manhattan'],
}
grid_search = GridSearchCV(knn_model,param_grid1, cv=4)
grid_search.fit(X, Y)
print("Meilleur score de validation croisée :", grid_search.best_score_)
print("Best hyperparameters:", grid_search.best_params_)

# Prédire les étiquettes sur l'ensemble de test
X_test = X_test.to_numpy() # Convertir X_test en tableau NumPy si X_test est une DataFrame Pandas
y_pred = knn_model.predict(X_test)
precision = precision_score(y_test,y_pred,average='weighted')
recall= recall_score(y_test,y_pred,average='weighted')

print("Précision: {:.2f}".format(precision))
print("Rappel: {:.2f}".format(recall))
#Présentez graphiquement les performances du modèle sur l'ensemble d'entraînement et l'ensemble de test en fonction des différentes valeurs de n_neighbors
# Définisser les valeurs de n_neighbors à évaluer
param_range = np.arange(1, 21)
# Calculer les performances pour l'ensemble d'entraînement et l'ensemble de test
train_scores, test_scores = validation_curve(
    KNeighborsClassifier(), X, Y, param_name="n_neighbors", param_range=param_range, cv=4
)
# Calculer la moyenne et l'écart-type des scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Tracer la courbe de validation
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean, label="Entraînement", color="blue", marker="o")
plt.fill_between(
    param_range,
    train_mean - train_std,
    train_mean + train_std,
    alpha=0.15,
    color="blue"
)
plt.plot(param_range, test_mean, label="Test", color="green", marker="o")
plt.fill_between(
    param_range,
    test_mean - test_std,
    test_mean + test_std,
    alpha=0.15,
    color="green"
)

# Ajouter des détails au graphique
plt.title("Courbe de Validation pour KNeighborsClassifier")
plt.xlabel("Nombre de voisins (n_neighbors)")
plt.ylabel("Score de classification")
plt.legend(loc="best")
plt.show()
#bech nhizo
conf_matrix_knn = confusion_matrix(y_test, y_pred)
print("Matrice de confusion pour kmean:\n", conf_matrix_knn)
#scatter knn
index = np.where(features == "worst area")[0][0]
plt.scatter(X_test[:,index],y_test,color='blue',label='worst area train')
plt.scatter(X_test[:,index],y_pred,color='red',label='worst area test')
plt.xlabel("worst area train")
plt.ylabel("worst area train")
plt.title("Prédictions par rapport à la colonne 'worst area'")
plt.legend()
plt.show()