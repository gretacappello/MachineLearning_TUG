
import os
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
import time
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
print("All good")
global PLOT_DIR, FINAL_PIPELINE, OUTLIER_DETECTOR, THRESHOLD, FEATURE_COLUMNS
global CLUSTER_SCALER, CLUSTERER, OUTLIER_MODELS, OUTLIER_THRESHOLDS, CLASSIFIERS
# Globals used by predict()
PLOT_DIR = None
FINAL_PIPELINE = None     # pipeline (scaler + classifier)
OUTLIER_DETECTOR = None   # classifier that predicts outlier probability
THRESHOLD = None          # threshold on detector proba to flag outlier
FEATURE_COLUMNS = None    # keep column order used at training time


def predict(X_test):
    """
    Predict labels and outliers for X_test.

    X_test: DataFrame with columns containing the same feature names used in training and 'id'
    returns:
      labels: np.ndarray of predicted class (0..3), shape (n_samples,)
      outliers: np.ndarray of 0/1 (inlier/outlier), shape (n_samples,)
    """
    global FINAL_PIPELINE, OUTLIER_DETECTOR, THRESHOLD, FEATURE_COLUMNS

    if FINAL_PIPELINE is None or OUTLIER_DETECTOR is None or THRESHOLD is None or FEATURE_COLUMNS is None:
        raise RuntimeError("Models not initialized. Train final pipeline + outlier detector in main() before calling predict().")

    X = X_test.copy()

    # ensure expected columns
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in test data: {missing}")

    X_feat = X[FEATURE_COLUMNS]

    # ---- Outlier prediction via density log-likelihood ----
    # OUTLIER_DETECTOR is a Pipeline (Scaler + GMM) or any model exposing score_samples
    ll = OUTLIER_DETECTOR.score_samples(X_feat)     # log p_theta(x)
    outliers = (ll <= THRESHOLD).astype(int)        # tau is in log-likelihood space

    # ---- Class prediction (only matters for inliers) ----
    labels = np.asarray(FINAL_PIPELINE.predict(X_feat)).ravel().astype(int)

    # Optional: for predicted outliers, label can be arbitrary since it's ignored.
    # Setting to 0 just avoids weird values.
    #labels[outliers == 1] = 0

    return labels, outliers

def generate_submission(test_data):
    label_predictions, outlier_predictions = predict(test_data)

    # ensure 1D arrays so pandas doesn't complain
    label_predictions = np.asarray(label_predictions).ravel()
    outlier_predictions = np.asarray(outlier_predictions).ravel()

    submission_df = pd.DataFrame({
        "id": test_data["id"].astype(int).to_numpy(),
        "label": label_predictions,
        "outlier": outlier_predictions
    })
    return submission_df

def make_EDA(X, y, X_out):
    global PLOT_DIR
    global CLUSTER_SCALER, CLUSTERER, OUTLIER_MODELS, OUTLIER_THRESHOLDS, CLASSIFIERS
    #Clustering



    # Scale data (use the SAME scaler as clustering)
    Xs = CLUSTER_SCALER.transform(X)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)

    # Cluster labels
    c_tr = CLUSTERER.predict(Xs)

    plt.figure(figsize=(7,6))
    sc = plt.scatter(Xp[:,0], Xp[:,1], c=c_tr, s=10, alpha=0.7)
    plt.colorbar(sc, label="Cluster ID")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("GMM Clustering (PCA projection)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot1.png"), dpi=300)
    plt.show()
    plt.close()


    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    Xt = tsne.fit_transform(Xs)
    c_tr = CLUSTERER.predict(Xs)

    plt.figure(figsize=(7,6))
    sc = plt.scatter(Xt[:,0], Xt[:,1], c=c_tr, s=10, alpha=0.7)
    plt.colorbar(sc, label="Cluster ID")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("GMM Clustering (t-SNE projection)")
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot2.png"), dpi=300)
    plt.show()
    plt.close()


    # Project BOTH datasets with PCA fitted on training data
    Xs_tr = CLUSTER_SCALER.transform(X)
    Xs_out = CLUSTER_SCALER.transform(X_out)

    Xp_tr = pca.fit_transform(Xs_tr)
    Xp_out = pca.transform(Xs_out)

    c_tr = CLUSTERER.predict(Xs_tr)
    c_out = CLUSTERER.predict(Xs_out)

    plt.figure(figsize=(7,6))
    plt.scatter(Xp_tr[:,0], Xp_tr[:,1], c=c_tr, s=10, alpha=0.4, label="Train")
    plt.scatter(Xp_out[:,0], Xp_out[:,1], color="red", s=40, marker="x", label="Outliers")
    plt.legend()
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("Clusters with Outlier Set Overlay (PCA)")
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot3.png"), dpi=300)
    plt.show()
    plt.close()


    centroids = CLUSTERER.means_
    centroids_df = pd.DataFrame(
        CLUSTER_SCALER.inverse_transform(centroids),
        columns=FEATURE_COLUMNS
    )

    # ---- Selected features ----
    centroids_df[["feature_0", "feature_1", "feature_2"]].plot(kind="bar", figsize=(8,4))
    plt.xlabel("Cluster")
    plt.ylabel("Centroid value")
    plt.title("Cluster centroids (selected features)")
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot4.png"), dpi=300)
    plt.show()
    plt.close()


    # ---- Top discriminative features ----
    feature_spread = centroids_df.std(axis=0)
    top_features = feature_spread.sort_values(ascending=False).head(8).index.tolist()

    print("Top discriminative features:", top_features)

    centroids_df[top_features].plot(kind="bar", figsize=(10,4))
    plt.xlabel("Cluster")
    plt.ylabel("Centroid value")
    plt.title("Cluster centroids (top discriminative features)")
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot5.png"), dpi=300)
    plt.show()
    plt.close()


    plt.figure(figsize=(10,3))
    plt.imshow(centroids_df.values, aspect="auto")
    plt.yticks(range(len(centroids_df)), [f"cluster {i}" for i in centroids_df.index])
    plt.xticks(range(len(centroids_df.columns)), centroids_df.columns, rotation=45, ha="right")
    plt.colorbar(label="Centroid value")
    plt.xlabel("Features")
    plt.ylabel("Clusters")
    plt.title("Cluster centroids across all features")
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot6.png"), dpi=300)
    plt.show()
    plt.close()

    probs = CLUSTERER.predict_proba(CLUSTER_SCALER.transform(X))
    max_prob = probs.max(axis=1)

    plt.figure(figsize=(7,6))
    plt.scatter(Xp[:,0], Xp[:,1], c=max_prob, s=10, cmap="viridis")
    plt.colorbar(label="Max cluster probability")
    plt.title("GMM confidence (PCA space)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot7.png"), dpi=300)
    plt.show()
    plt.close()

    # Build crosstab (normalized)
    ct = pd.crosstab(
        c_tr,
        y.reset_index(drop=True),
        normalize="index"
    )

    # Plot
    ct.plot(
        kind="bar",
        stacked=True,
        figsize=(7,4),
        colormap="tab10"
    )

    plt.xlabel("Cluster")
    plt.ylabel("Fraction of samples")
    plt.title("Label distribution within each cluster")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "cluster_plot8.png"), dpi=300)
    plt.show()
    plt.close()

    #PLOT 0
    features = ["feature_0", "feature_1", "feature_2", "feature_3"]

    pairs = list(itertools.combinations(features, 2))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, (f1, f2) in zip(axes, pairs):
        ax.scatter(X[f1], X[f2], alpha=0.2, label="D", s=10)
        ax.scatter(X_out[f1], X_out[f2], color="red", label="Outliers", s=30)
        ax.set_xlabel(f1)
        ax.set_ylabel(f2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "test_correl.png"), dpi=300)
    plt.show()
    plt.close()

    #Plot 1: class label distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=y)
    plt.title("Distribution of Class Labels")
    plt.xlabel("Class label")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"), dpi=300)
    plt.show()
    plt.close()


    #Plot 2:Feature distributions (inliers vs outliers)
    features_to_plot = ["feature_0", "feature_1", "feature_2"]

    fig, axes = plt.subplots(1, 3, figsize=(15,4))
    for i, f in enumerate(features_to_plot):
        #sns.kdeplot(X[f], label="D (mixed)", ax=axes[i])
        #sns.kdeplot(X_out[f], label="Known outliers", ax=axes[i])

        sns.kdeplot(X[f], label="D (mixed)", ax=axes[i], clip=(-30, 30))
        sns.kdeplot(X_out[f], label="Known outliers", ax=axes[i], linestyle="--")

        axes[i].set_title(f"Distribution of {f}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_distributions_inliers_vs_outliers.png"), dpi=300)
    plt.show()
    plt.close()

    #Plot 3: FULL feature correlation heatmap
    plt.figure(figsize=(10,8))
    corr = X.corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0.95,      # emphasize differences
        vmin=0.9,
        vmax=1.0
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_correlation_heatmap.png"), dpi=300)
    plt.show()
    plt.close()

    #Plot 4: PCA (colored by class)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6,5))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y,
        cmap="tab10",
        alpha=0.6
    )

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Projection Colored by Class Label")
    plt.legend(*scatter.legend_elements(), title="Class")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pca_by_class.png"), dpi=300)
    plt.show()
    plt.close()

    #Plot 5 – PCA: Inliers vs Known Outliers
    X_out_scaled = scaler.transform(X_out)
    X_out_pca = pca.transform(X_out_scaled)

    plt.figure(figsize=(6,5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.4, label="D (mixed)")
    plt.scatter(X_out_pca[:, 0], X_out_pca[:, 1], color="red", label="Known outliers")

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA: Training Data vs Known Outliers")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "pca_inliers_vs_outliers.png"), dpi=300)
    plt.show()
    plt.close()

    #Plot 6: Feature variance
    feature_variance = X.var().sort_values(ascending=False)

    plt.figure(figsize=(8,4))
    sns.barplot(x=feature_variance.index, y=feature_variance.values)
    plt.xticks(rotation=45)
    plt.ylabel("Variance")
    plt.title("Feature Variance")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "feature_variance.png"), dpi=300)
    plt.show()
    plt.close()


def make_scaled_pca_clf(clf, n_components, random_state=42):
    """
    StandardScaler -> PCA -> clf
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
        ("clf", clf),
    ])


def build_experiments(random_state=42):
    experiments = []

   
    experiments.append(("LR_pca3_C1_bal", make_scaled_pca_clf(
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", solver="lbfgs"),
        n_components=3, random_state=random_state
    )))

    experiments.append(("LR_pca5_C1_bal", make_scaled_pca_clf(
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", solver="lbfgs"),
        n_components=5, random_state=random_state
    )))

    experiments.append(("LR_pca9_C1_bal", make_scaled_pca_clf(
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", solver="lbfgs"),
        n_components=9, random_state=random_state
    )))


    experiments.append(("SVM_rbf_pca3_C5", make_scaled_pca_clf(
        SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced"),
        n_components=3, random_state=random_state
    )))

    experiments.append(("SVM_rbf_pca5_C5", make_scaled_pca_clf(
        SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced"),
        n_components=5, random_state=random_state
    )))

    experiments.append(("SVM_rbf_pca9_C5", make_scaled_pca_clf(
        SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, class_weight="balanced"),
        n_components=9, random_state=random_state
    )))

    experiments.append(("SVM_rbf_pca3_C5_cal", CalibratedClassifierCV(
    estimator=make_scaled_pca_clf(SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"),
                                  n_components=3, random_state=random_state),method="sigmoid",cv=3
    )))

    experiments.append(("SVM_rbf_pca5_C5_cal", CalibratedClassifierCV(
    estimator=make_scaled_pca_clf(SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"),
                                  n_components=5, random_state=random_state),method="sigmoid",cv=3
    )))

    experiments.append(("SVM_rbf_pca9_C5_cal", CalibratedClassifierCV(
    estimator=make_scaled_pca_clf(SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced"),
                                  n_components=9, random_state=random_state),method="sigmoid",cv=3
    )))

    experiments.append(("KNN_pca5_K21_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=21, weights="distance"),
        n_components=5, random_state=random_state
    )))

    experiments.append(("KNN_pca5_K5_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
        n_components=5, random_state=random_state
    )))

    experiments.append(("KNN_pca5_K11_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=11, weights="distance"),
        n_components=9, random_state=random_state
    )))

    experiments.append(("KNN_pca9_K21_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=21, weights="distance"),
        n_components=9, random_state=random_state
    )))

    experiments.append(("KNN_pca9_K5_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=5, weights="distance"),
        n_components=9, random_state=random_state
    )))

    experiments.append(("KNN_pca9_K11_dist", make_scaled_pca_clf(
        KNeighborsClassifier(n_neighbors=11, weights="distance"),
        n_components=9, random_state=random_state
    )))



    experiments.append(("RF_500_depth10_bal_sub", RandomForestClassifier(
    n_estimators=500, max_depth=10, class_weight="balanced_subsample",
    random_state=random_state, n_jobs=-1
    )))

    experiments.append(("RF_400_depth15_bal_sub", RandomForestClassifier(
    n_estimators=400, max_depth=15, class_weight="balanced_subsample",
    random_state=random_state, n_jobs=-1
    )))

    experiments.append(("RF_600_depth15_bal_sub", RandomForestClassifier(
    n_estimators=600, max_depth=15, class_weight="balanced_subsample",
    random_state=random_state, n_jobs=-1
    )))


    w2 = {
    0: 1.3,   # slightly up
    1: 0.6,   # down (dominant class)
    2: 1.1,   # mild up
    3: 1.6    # strong up (weakest class)
    }

    experiments.append(("RF_600_depth10_w2", RandomForestClassifier(
    n_estimators=600,
    max_depth=10,
    min_samples_leaf=10,
    class_weight=w2,
    random_state=42,
    n_jobs=-1
    )))


    experiments.append(("RF_600_depth15_minsamplw2", RandomForestClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_leaf=10,
    class_weight=w2,
    random_state=42,
    n_jobs=-1
    )))


    experiments.append(("ET_400_depthNone", ExtraTreesClassifier(
        n_estimators=400, max_depth=None,random_state=random_state, n_jobs=-1
    )))

    experiments.append(("ET_800_depthNone", ExtraTreesClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_leaf=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=random_state,
    n_jobs=-1
    )))

    experiments.append(("ET_500_depth12", ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=random_state,
    n_jobs=-1
    )))

    experiments.append(("ET_400_depthNonew2", ExtraTreesClassifier(
        n_estimators=400, max_depth=None, class_weight=w2,random_state=random_state, n_jobs=-1
    )))

    experiments.append(("ET_800_depthNonew2", ExtraTreesClassifier(
    n_estimators=800,
    max_depth=None,
    class_weight=w2,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=random_state,
    n_jobs=-1
    )))

    experiments.append(("ET_500_depth12w2", ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    class_weight=w2,
    min_samples_leaf=10,
    random_state=random_state,
    n_jobs=-1
    )))

    experiments.append(("ET_400_depthNone_bal_sub", ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )))

    experiments.append(("ET_800_depthNone_bal_sub", ExtraTreesClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )))

    experiments.append(("HGB_lr0.05_depth6_iter600", HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6, max_iter=600, random_state=random_state
    )))

    experiments.append(("LGBM_learn0.05_700", LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, random_state=random_state, n_jobs=1
    )))

    experiments.append(("CatBoost_depth6_lr0.05", CatBoostClassifier(
        loss_function="MultiClass",
        iterations=1500, learning_rate=0.05, depth=6,
        l2_leaf_reg=3.0, random_seed=random_state, verbose=False
    )))

    experiments.append(("XGB_depth6_lr005", XGBClassifier(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )))
    experiments.append(("LDA", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ])))

    experiments.append(("LDA_shrinkage_auto", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    ])))

    experiments.append(("QDA_reg0.1", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", QuadraticDiscriminantAnalysis(reg_param=0.1))
    ])))

    experiments.append(("MLP_64x32_relu", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64,32),
                            activation="relu",
                            alpha=1e-3,
                            learning_rate_init=1e-3,
                            max_iter=2000,
                            random_state=random_state))
    ])))

    return experiments


def evaluate_experiment(name, model, X, y, cv):
    """
    Runs cross_val_predict for a model/pipeline and returns metrics in a dict.
    """
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True, digits=4)

    result = {"experiment": name, "accuracy": acc, "macro_f1": report["macro avg"]["f1-score"]}

    for cls in sorted(y.unique()):
        cls_str = str(cls)
        result[f"precision_{cls}"] = report[cls_str]["precision"]
        result[f"recall_{cls}"] = report[cls_str]["recall"]
        result[f"f1_{cls}"] = report[cls_str]["f1-score"]

    return result


def run_experiments(experiments, X, y, cv, save_path=None, tag=""):
    results = []
    for name, model in experiments:
        exp_name = f"{tag}{name}" if tag else name
        res = evaluate_experiment(exp_name, model, X, y, cv)
        results.append(res)
        print(f"Done: {exp_name} | acc={res['accuracy']:.4f} | macro_f1={res['macro_f1']:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="macro_f1", ascending=False)
    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Saved results to {save_path}")
    return results_df


def tune_random_forest(X, y, cv, n_iter=40, random_state=42):
    
    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 8, 12, 15, 20],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5],
        "class_weight": [None, "balanced", "balanced_subsample"]
    }
    """
    param_dist = {
        "n_estimators": [400],
        "max_depth": [15],
        "min_samples_split": [8],
        "min_samples_leaf": [4],
        "max_features": [0.5],
        "class_weight": ["balanced_subsample"]
    }
    """
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rs = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter,
        scoring="f1_macro", cv=cv, random_state=random_state, n_jobs=-1, verbose=1
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"RandomizedSearchCV done in {time.time()-t0:.0f}s")
    print("Best RF params:", rs.best_params_)
    print("Best RF CV f1_macro:", rs.best_score_)

    return rs.best_estimator_



def tune_ET(X, y, cv):

    et =  ExtraTreesClassifier(random_state=42, n_jobs=1)  # n_jobs=1 to avoid nested parallelism


    w3 = {
        0: 1.1,
        1: 0.8,
        2: 1.1,
        3: 1.6
    }

    param_dist_et = {
        "n_estimators": [400, 600, 800, 1000],
        "max_depth": [None, 12, 15],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4, 6],
        "max_features": ["sqrt", 0.5, 0.7],
        "class_weight": ["balanced", "balanced_subsample", None, w3]
    }
    
    rs_et = RandomizedSearchCV(
        et, param_distributions=param_dist_et, n_iter=40,
        scoring="f1_macro", cv=cv, random_state=42, n_jobs=-1, verbose=1
    )
    t0 = time.time()
    rs_et.fit(X, y)
    print(f"RandomizedSearchCV done in {time.time()-t0:.0f}s")
    best_et = rs_et.best_estimator_
    print("Best ET params:", rs_et.best_params_)
    print("Best ET CV macro-F1:", rs_et.best_score_)
    return rs_et.best_estimator_



def train_gmm_density(X, n_components=10, random_state=42):
    """
    Train a multi-modal density model p_theta(x) using a GMM.
    X may contain inliers and outliers.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state
        ))
    ])
    model.fit(X)
    return model

"""
def choose_threshold_ll(model, X_out, target_recall=0.90):
    
    #Choose tau such that target_recall of known outliers
    #have log p_theta(x) <= tau.
    
    ll_out = model.score_samples(X_out)
    tau = np.quantile(ll_out, 1.0 - target_recall)
    return tau

def filter_outliers(model, X, tau):
    
    #Returns mask: True = inlier, False = outlier
    
    ll = model.score_samples(X)
    return ll > tau
"""

def select_and_build_final_model(results_df, experiments):
    """
    Given results DataFrame (sorted desc by macro_f1) and list of (name, model),
    pick the best experiment, take the corresponding model, and build a pipeline:
     - if the chosen model is already a Pipeline -> use it directly
     - else wrap with StandardScaler -> Pipeline([("scaler", StandardScaler()), ("clf", model)])
    Returns (best_name, pipeline_estimator)
    """
    # results_df is expected to be sorted descending by macro_f1
    best_row = results_df.iloc[0]
    best_exp = best_row["experiment"]
    # strip optional prefixes
    key = best_exp.replace("FULL_", "").replace("INLIERS_", "")
    name_to_model = {name: model for (name, model) in experiments}
    if key not in name_to_model:
        # fallback: pick highest from experiments list (by index) - unlikely
        chosen = experiments[0][1]
        chosen_name = experiments[0][0]
    else:
        chosen = name_to_model[key]
        chosen_name = key

    # If chosen is a Pipeline, use as-is; otherwise wrap with a scaler
    if isinstance(chosen, Pipeline):
        final_pipeline = chosen
    else:
        final_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", chosen)])

    print(f"Selected final model: {chosen_name}")
    return chosen_name, final_pipeline

def proxy_outlier_f1_from_confident_inliers(density_model, tau, X_D, X_out, q_conf=0.80):
    """
    Proxy F1 where negatives are chosen as high-confidence inliers
    (top (1-q_conf)% by likelihood in D).
    Positives are known outliers from D_out.
    """
    ll_D = density_model.score_samples(X_D)
    conf_mask = ll_D >= np.quantile(ll_D, q_conf)
    X_neg = X_D[conf_mask]                 # assumed inliers (high confidence)
    X_pos = X_out                          # known outliers

    ll_neg = density_model.score_samples(X_neg)
    ll_pos = density_model.score_samples(X_pos)

    y_true = np.concatenate([
        np.zeros(len(ll_neg), dtype=int),
        np.ones(len(ll_pos), dtype=int)
    ])
    y_pred = np.concatenate([
        (ll_neg <= tau).astype(int),
        (ll_pos <= tau).astype(int)
    ])

    return f1_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")

def fit_clusterer(X_df, n_clusters=3, random_state=42):
    """
    Fit scaler + clustering GMM on D (mixed).
    Returns: (scaler, clusterer, cluster_ids)
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df)

    clusterer = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state
    )
    clusterer.fit(Xs)

    c = clusterer.predict(Xs)
    return scaler, clusterer, c

def assign_clusters(scaler, clusterer, X_df):
    Xs = scaler.transform(X_df)
    return clusterer.predict(Xs)



def main():
    global PLOT_DIR, FINAL_PIPELINE, OUTLIER_DETECTOR, THRESHOLD, FEATURE_COLUMNS
    global CLUSTER_SCALER, CLUSTERER, OUTLIER_MODELS, OUTLIER_THRESHOLDS, CLASSIFIERS

    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load datasets
    D = pd.read_csv("D.csv") #mix set
    D_out = pd.read_csv("D_out.csv") #only outlier set

    X = D.drop(columns=["id", "label"]).reset_index(drop=True)
    y = D["label"].reset_index(drop=True)

    # Features from training
    FEATURE_COLUMNS = list(X.columns)

    X_out = D_out.drop(columns=["id"]).reset_index(drop=True)
    X_train_full, X_va, y_train_full, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_out_cal, X_out_eval = train_test_split(
        X_out, test_size=0.15, random_state=42
    )

    

    # 1) Fit clusterer on D
    CLUSTER_SCALER, CLUSTERER, c_train = fit_clusterer(X, n_clusters=3, random_state=42)
    c_out = assign_clusters(CLUSTER_SCALER, CLUSTERER, X)

    # after fitting clusterer on X_tr
    c_tr = CLUSTERER.predict(CLUSTER_SCALER.transform(X))
    sil = silhouette_score(CLUSTER_SCALER.transform(X), c_tr)
    print("silhouette:", sil)

    # distribution of clusters
    print("cluster sizes:", np.bincount(c_tr))


    #############################################
    # (a) Exploratory Data Analysis [6 points]  #
    #############################################

    #make_EDA(X_train_full, y_train_full, X_out_cal)


    ##########################################
    #  (b)   Baseline Model [3 points]       #
    ##########################################

    # Validation (cross-val) set-up
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": list(range(1, 31, 2)),
        "knn__weights": ["uniform", "distance"]
    }

    grid = GridSearchCV(
        pipe_knn,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1
    )

    grid.fit(X_train_full, y_train_full)

    print("Chosen K (baseline):", grid.best_params_["knn__n_neighbors"])
    print("Best CV macro-F1 (baseline):", grid.best_score_)

    # Freeze best model
    best_knn = grid.best_estimator_

    # Evaluate ONCE on validation set
    y_pred_va = best_knn.predict(X_va)

    print("\nValidation Accuracy:", accuracy_score(y_va, y_pred_va))
    print("\nValidation Classification Report:")
    print(classification_report(y_va, y_pred_va, digits=4))
    #return


    #############################################################
    #  (c) Model experimentation and validation [19 points]     #
    #############################################################

    # Run candidate all experiments (on training set)
    #experiments = build_experiments(random_state=42)
    #results_full = run_experiments(experiments, X_train_full, y_train_full, cv, save_path="experiment_results_full.csv", tag="FULL_")
    #print("\nTop FULL results:")
    #print(results_full[["experiment", "accuracy", "macro_f1"]].head(10))
    #print("Finished testing all experiments on all dataset")

    #Train Random Forest with tuning all all the dataset
    #print("\nTuning RandomForest X_train_full...")
    #best_rf_in = tune_random_forest(X_train_full, y_train_full, cv=cv, n_iter=40, random_state=42)
    #FINAL_PIPELINE = Pipeline([("clf", best_rf_in)])

    #print("\nTuning RandomForest on D (mix inlier and outlier)...")
    #best_rf_in = tune_random_forest(X_train_full, y_train_full, cv=cv, n_iter=40, random_state=42)

    #FINAL_PIPELINE = Pipeline([("clf", best_rf_in)])
    #print("Final pipeline trained on inliers-only data (tuned on Din).")
    #y_pred_cv = cross_val_predict(
    #    FINAL_PIPELINE,
    #    X_train_full,
    #    y_train_full,
    #    cv=cv,
    #    n_jobs=-1
    #)

    #acc = accuracy_score(y_train_full, y_pred_cv)
    #print(f"\nRF Accuracy (CV, on D): {acc:.4f}\n")

    #print("RF Classification Report (CV, on D):")
    #print(classification_report(y_train_full, y_pred_cv))

    #FINAL_PIPELINE.fit(X_train_full, y_train_full)

    #scores = cross_val_score(FINAL_PIPELINE, X_train_full, y_train_full, scoring="f1_macro", cv=cv)
    #f1_macro_mean = scores.mean()
    #print("CV macro-F1: %.3f ± %.3f" % (scores.mean(), scores.std()))
    #print("Finished tuning RF on all dataset")


    #############################################################
    #  (d) Outlier Detection [7 points]                         #
    #############################################################

    best_f1_global = -1.0
    best_K, best_cov, best_tau = None, None, None
    best_model = None

    # optional: set reasonable bounds for how many points in D you allow to be flagged as outliers
    MAX_REMOVAL = 0.35   # don't remove more than 35% of D
    MIN_REMOVAL = 0.01   # don't remove less than 1% (otherwise detector does nothing)

    q_conf = 0.90  # try 0.90 or 0.95

    for K in [3, 4, 5]:
        for cov in ["diag", "full"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("gmm", GaussianMixture(
                    n_components=K,
                    covariance_type=cov,
                    reg_covar=1e-5,
                    random_state=42
                ))
            ])
            model.fit(X_train_full[FEATURE_COLUMNS])

            # scores on D and calibration outliers
            ll_D = model.score_samples(X_train_full[FEATURE_COLUMNS])
            ll_out_cal = model.score_samples(X_out_cal)

            # high-confidence inliers as proxy negatives (from D)
            ll_neg = ll_D[ll_D >= np.quantile(ll_D, q_conf)]

            # candidate thresholds (taus) from combined range
            grid = np.quantile(
                np.concatenate([ll_neg, ll_out_cal]),
                np.linspace(0.01, 0.99, 200)
            )

            # proxy classification set for tuning tau
            y_true = np.concatenate([
                np.zeros(len(ll_neg), dtype=int),
                np.ones(len(ll_out_cal), dtype=int)
            ])
            ll_all = np.concatenate([ll_neg, ll_out_cal])

            best_tau_local, best_f1_local = None, -1.0

            for tau in grid:
                # CONSTRAINT: avoid degenerate taus that remove almost everything (kills precision)
                frac_removed = float(np.mean(ll_D <= tau))
                if frac_removed > MAX_REMOVAL or frac_removed < MIN_REMOVAL:
                    continue

                y_pred = (ll_all <= tau).astype(int)   # outlier if ll <= tau
                f1 = f1_score(y_true, y_pred)          # binary F1 for outliers
                if f1 > best_f1_local:
                    best_f1_local, best_tau_local = f1, tau

            # if all taus were filtered out by constraints, fall back to a safe tau (e.g. 10% removal)
            if best_tau_local is None:
                best_tau_local = np.quantile(ll_D, 0.10)
                best_f1_local = -1.0  # indicates fallback was used

            # quick sanity check on held-out outliers (not used for tau tuning)
            ll_out_eval = model.score_samples(X_out_eval)
            recall_eval = float(np.mean(ll_out_eval <= best_tau_local))
            frac_removed_final = float(np.mean(ll_D <= best_tau_local))

            print(f"K={K}, cov={cov}, proxyF1={best_f1_local:.3f}, "
                f"removed(D)={frac_removed_final:.2%}, recall_out_eval={recall_eval:.2%}")

            # pick best model (primary: proxy F1; tie-breaker: higher recall on eval outliers)
            if (best_f1_local > best_f1_global) or (best_f1_local == best_f1_global and recall_eval > 0.0):
                best_f1_global = best_f1_local
                best_K, best_cov, best_tau = K, cov, best_tau_local
                best_model = model

    # FIX: now we "fix" them once, AFTER the search
    DENSITY_MODEL = best_model
    TAU = best_tau

    print(f"\nSelected density model: K={best_K}, cov={best_cov}, TAU={float(TAU):.4f}, proxyF1={best_f1_global:.3f}")

    # Extra sanity prints you should keep
    ll_D_final = DENSITY_MODEL.score_samples(X_train_full[FEATURE_COLUMNS])
    ll_out_final = DENSITY_MODEL.score_samples(X_out[FEATURE_COLUMNS])
    print("Recall on D_out (all):", float(np.mean(ll_out_final <= TAU)))
    print("Fraction removed from D:", float(np.mean(ll_D_final <= TAU)))
    # outlier flags for X_train_full
    is_out = (DENSITY_MODEL.score_samples(X_train_full[FEATURE_COLUMNS]) <= TAU)

    # cluster IDs for the SAME X_train_full
    Xs_tr = CLUSTER_SCALER.transform(X_train_full[FEATURE_COLUMNS])
    c_tr = CLUSTERER.predict(Xs_tr)
    is_out_train = is_out
    print("len(is_out) =", len(is_out))
    print("len(c_tr)   =", len(c_tr))

    df_diag = pd.DataFrame({"cluster": c_tr, "is_out": is_out})

    rates = df_diag.groupby("cluster")["is_out"].mean()

    counts = df_diag.groupby("cluster")["is_out"].agg(["count", "sum"])
    counts = counts.rename(columns={"count": "n_total", "sum": "n_outliers"})
    counts["outlier_rate"] = counts["n_outliers"] / counts["n_total"]

    print(counts)

    plt.figure(figsize=(8,4))
    counts["outlier_rate"].plot(kind="bar", figsize=(6,3))
    plt.xlabel("Cluster")
    plt.ylabel("Outlier rate")
    plt.title("Outlier rate by cluster (using global TAU)")
    plt.savefig(os.path.join(PLOT_DIR, "outlierRate_GLOBAL_TAUiteration1.png"), dpi=300)
    plt.show()
    plt.close()


    X_in = X_train_full.loc[~is_out_train].reset_index(drop=True)
    y_in = y_train_full.loc[~is_out_train].reset_index(drop=True)


    for K in [3, 4, 5]:
        for cov in ["diag", "full"]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("gmm", GaussianMixture(
                    n_components=K,
                    covariance_type=cov,
                    reg_covar=1e-5,
                    random_state=42
                ))
            ])
            model.fit(X_in[FEATURE_COLUMNS])

            # scores on D and calibration outliers
            ll_D = model.score_samples(X_in[FEATURE_COLUMNS])
            ll_out_cal = model.score_samples(X_out_cal)

            # high-confidence inliers as proxy negatives (from D)
            ll_neg = ll_D[ll_D >= np.quantile(ll_D, q_conf)]

            # candidate thresholds (taus) from combined range
            grid = np.quantile(
                np.concatenate([ll_neg, ll_out_cal]),
                np.linspace(0.01, 0.99, 200)
            )

            # proxy classification set for tuning tau
            y_true = np.concatenate([
                np.zeros(len(ll_neg), dtype=int),
                np.ones(len(ll_out_cal), dtype=int)
            ])
            ll_all = np.concatenate([ll_neg, ll_out_cal])

            best_tau_local, best_f1_local = None, -1.0

            for tau in grid:
                # CONSTRAINT: avoid degenerate taus that remove almost everything (kills precision)
                frac_removed = float(np.mean(ll_D <= tau))
                if frac_removed > MAX_REMOVAL or frac_removed < MIN_REMOVAL:
                    continue

                y_pred = (ll_all <= tau).astype(int)   # outlier if ll <= tau
                f1 = f1_score(y_true, y_pred)          # binary F1 for outliers
                if f1 > best_f1_local:
                    best_f1_local, best_tau_local = f1, tau

            # if all taus were filtered out by constraints, fall back to a safe tau (e.g. 10% removal)
            if best_tau_local is None:
                best_tau_local = np.quantile(ll_D, 0.10)
                best_f1_local = -1.0  # indicates fallback was used

            # quick sanity check on held-out outliers (not used for tau tuning)
            ll_out_eval = model.score_samples(X_out_eval)
            recall_eval = float(np.mean(ll_out_eval <= best_tau_local))
            frac_removed_final = float(np.mean(ll_D <= best_tau_local))

            print(f"K={K}, cov={cov}, proxyF1={best_f1_local:.3f}, "
                f"removed(D)={frac_removed_final:.2%}, recall_out_eval={recall_eval:.2%}")

            # pick best model (primary: proxy F1; tie-breaker: higher recall on eval outliers)
            if (best_f1_local > best_f1_global) or (best_f1_local == best_f1_global and recall_eval > 0.0):
                best_f1_global = best_f1_local
                best_K, best_cov, best_tau = K, cov, best_tau_local
                best_model = model

    # FIX: now we "fix" them once, AFTER the search
    DENSITY_MODEL = best_model
    TAU = best_tau

    print(f"\nSelected density model: K={best_K}, cov={best_cov}, TAU={float(TAU):.4f}, proxyF1={best_f1_global:.3f}")

    # Extra sanity prints you should keep
    ll_D_final = DENSITY_MODEL.score_samples(X_train_full[FEATURE_COLUMNS])
    ll_out_final = DENSITY_MODEL.score_samples(X_out[FEATURE_COLUMNS])
    print("Recall on D_out (all):", float(np.mean(ll_out_final <= TAU)))
    print("Fraction removed from D:", float(np.mean(ll_D_final <= TAU)))
    # outlier flags for X_train_full
    is_out = (DENSITY_MODEL.score_samples(X_train_full[FEATURE_COLUMNS]) <= TAU)

    # cluster IDs for the SAME X_train_full
    Xs_tr = CLUSTER_SCALER.transform(X_train_full[FEATURE_COLUMNS])
    c_tr = CLUSTERER.predict(Xs_tr)
    is_out_train = is_out
    print("len(is_out) =", len(is_out))
    print("len(c_tr)   =", len(c_tr))

    df_diag = pd.DataFrame({"cluster": c_tr, "is_out": is_out})

    rates = df_diag.groupby("cluster")["is_out"].mean()

    counts = df_diag.groupby("cluster")["is_out"].agg(["count", "sum"])
    counts = counts.rename(columns={"count": "n_total", "sum": "n_outliers"})
    counts["outlier_rate"] = counts["n_outliers"] / counts["n_total"]

    print(counts)

    plt.figure(figsize=(8,4))
    counts["outlier_rate"].plot(kind="bar", figsize=(6,3))
    plt.xlabel("Cluster")
    plt.ylabel("Outlier rate")
    plt.title("Outlier rate by cluster (using global TAU)")
    plt.savefig(os.path.join(PLOT_DIR, "outlierRate_GLOBAL_TAU_iteration2.png"), dpi=300)
    plt.show()
    plt.close()


    X_in = X_train_full.loc[~is_out_train].reset_index(drop=True)
    y_in = y_train_full.loc[~is_out_train].reset_index(drop=True)

    #############################################################
    #  (e) Leaderboard Predictions [+ Bonus points]             #
    ############################################################


    """
    results_full = run_experiments(experiments, X_in[FEATURE_COLUMNS], y_in, cv, save_path="experiment_results_inliers.csv", tag="INLIERS_")
    print("\nTop INLIERS results:")
    print(results_full[["experiment", "accuracy", "macro_f1"]].head(10))
    print("Finished testing all experiments on ONLY INLIERS dataset")
    """

    #print("\nTuning RandomForest on Din (inliers only)...")
    #best_rf_in = tune_random_forest(X_in[FEATURE_COLUMNS], y_in, cv=cv, n_iter=40, random_state=42)
    #print("Finished tuning RF on ONLY INLIERS dataset")

    print("\nTuning ET on Din (inliers only)...")
    best_et_in = tune_ET(X_in[FEATURE_COLUMNS], y_in, cv=cv)
    print("Finished tuning ET on ONLY INLIERS dataset")

    FINAL_PIPELINE = Pipeline([("clf", best_et_in)])
    
    y_pred_cv = cross_val_predict(
        FINAL_PIPELINE,
        X_in[FEATURE_COLUMNS],
        y_in,
        cv=cv,
        n_jobs=-1
    )

    acc = accuracy_score(y_in, y_pred_cv)
    print(f"\nET Accuracy (CV, on Din): {acc:.4f}\n")

    print("ET Classification Report (CV, on Din):")
    print(classification_report(y_in, y_pred_cv))

    FINAL_PIPELINE.fit(X_in[FEATURE_COLUMNS], y_in)
    print("Final pipeline trained on inliers-only data (tuned on Din).")

    scores = cross_val_score(FINAL_PIPELINE, X_in[FEATURE_COLUMNS], y_in, scoring="f1_macro", cv=cv)
    f1_macro_mean = scores.mean()
    print("CV macro-F1: %.3f ± %.3f" % (scores.mean(), scores.std()))

    print("Finished with training")

    print("Starting evalutation of score f1 macro outlier detections")
    f1_bin, f1_macro = proxy_outlier_f1_from_confident_inliers(
        DENSITY_MODEL,
        TAU,
        X_train_full[FEATURE_COLUMNS],
        X_out[FEATURE_COLUMNS],
        q_conf=0.90
    )
    print("Proxy Outlier F1 (binary) detector outliers:", f1_bin)
    print("Proxy Outlier F1 (macro) detector outliers:", f1_macro)


    OUTLIER_DETECTOR = DENSITY_MODEL   # keep your global name if you want
    THRESHOLD = TAU                   # threshold is now log-likelihood tau


    # compute synthetic leaderboard Score (adjust weights if assignment specifies)
    ClassScore = f1_macro_mean * 100.0
    OutlierScore = f1_macro * 100.0
    CombinedScore = 0.8 * ClassScore + 0.2 * OutlierScore
    print(f"\nSimulated Leaderboard Scores -> Score: {CombinedScore:.2f}, ClassScore: {ClassScore:.2f}, OutlierScore: {OutlierScore:.2f}")


    # Persist artifacts
    #joblib.dump(FINAL_PIPELINE, "final_pipeline.joblib")
    #joblib.dump(OUTLIER_DETECTOR, "outlier_detector.joblib")
    #joblib.dump(THRESHOLD, "threshold.joblib")
    #joblib.dump(FEATURE_COLUMNS, "feature_columns.joblib")
    #print("Saved final_pipeline.joblib, outlier_detector.joblib, threshold.joblib, feature_columns.joblib")

    # Create submission files using trained final pipeline and detector
    GROUPNAME = "SimplyTheBest"
    df_leaderboard = pd.read_csv("D_test_leaderboard.csv")
    submission_df_lb = generate_submission(df_leaderboard)
    submission_df_lb.to_csv(f"submission_leaderboard_{GROUPNAME}.csv", index=False)

    df_final = pd.read_csv("D_test_final.csv")
    submission_df_final = generate_submission(df_final)
    submission_df_final.to_csv(f"submission_final_{GROUPNAME}.csv", index=False)

    print("Saved submission files:")
    print(f"submission_leaderboard_{GROUPNAME}.csv")
    print(f"submission_final_{GROUPNAME}.csv")


if __name__ == "__main__":
    main()
