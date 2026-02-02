
import os
import itertools
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.base import clone
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
global CLUSTER_SCALER, CLUSTERER, OUTLIER_MODELS, OUTLIER_THRESHOLDS, CLASSIFIERS, TAU_BY_CLUSTER, LABEL_FP
from collections import Counter
from scipy.stats import loguniform, randint, uniform

PLOT_DIR = None
FINAL_PIPELINE = None     # pipeline (scaler + classifier)
OUTLIER_DETECTOR = None   # classifier that predicts outlier probability
THRESHOLD = None          # threshold on detector proba to flag outlier
FEATURE_COLUMNS = None    # keep column order used at training time
CLUSTER_MODELS = None     # dict: {cluster_id: sklearn Pipeline}
FALLBACK_MODEL = None     # Pipeline for safety (if cluster missing / too small)

def predict(X_test):
    """
    Predict labels and outliers for X_test.
    """
    global FINAL_PIPELINE, OUTLIER_DETECTOR, THRESHOLD, FEATURE_COLUMNS
    global CLUSTER_SCALER, CLUSTERER, CLUSTER_MODELS, FALLBACK_MODEL, TAU_BY_CLUSTER

    if (FINAL_PIPELINE is None) and (CLUSTER_MODELS is None):
        raise RuntimeError("No classifier available: both FINAL_PIPELINE and CLUSTER_MODELS are None.")
    if OUTLIER_DETECTOR is None or THRESHOLD is None or FEATURE_COLUMNS is None:
        raise RuntimeError("Outlier detector / threshold / feature columns not initialized.")

    X = X_test.copy()

    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns in test data: {missing}")

    X_feat = X[FEATURE_COLUMNS]

    # ---- Outlier prediction via density log-likelihood ----
    #ll = OUTLIER_DETECTOR.score_samples(X_feat)
    #outliers = (ll <= THRESHOLD).astype(int)
    ll = OUTLIER_DETECTOR.score_samples(X_feat)  # log p_theta(x)
    outliers = (ll <= THRESHOLD).astype(int)  # tau is in log-likelihood space

    # ---- Class prediction ----
    # If cluster-specific models are available, route by cluster id
    if LABEL_FP == True:
        labels = np.asarray(FINAL_PIPELINE.predict(X_feat)).ravel().astype(int)
        print(f"Labels created using {FINAL_PIPELINE}")
    else:
        if CLUSTER_MODELS is not None:
            if CLUSTER_SCALER is None or CLUSTERER is None:
                raise RuntimeError("Clusterer not initialized but CLUSTER_MODELS is set.")

            Xs = CLUSTER_SCALER.transform(X_feat)
            c = CLUSTERER.predict(Xs)

            labels = np.zeros(len(X_feat), dtype=int)
            for k in np.unique(c):
                idx = np.where(c == k)[0]
                model_k = CLUSTER_MODELS.get(int(k), None)
                if model_k is None:
                    if FALLBACK_MODEL is None:
                        raise RuntimeError(f"No model for cluster {k} and FALLBACK_MODEL is None.")
                    labels[idx] = np.asarray(FALLBACK_MODEL.predict(X_feat.iloc[idx])).ravel().astype(int)
                else:
                    labels[idx] = np.asarray(model_k.predict(X_feat.iloc[idx])).ravel().astype(int)
                    print(f"Labels created using {model_k}")
        else:
            labels = np.asarray(FINAL_PIPELINE.predict(X_feat)).ravel().astype(int) #security check!!

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


def plot_pca_cluster_vs_label(X, y, cluster_scaler, clusterer, feature_cols):
    Xs = cluster_scaler.transform(X[feature_cols])
    c = clusterer.predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)

    plt.figure(figsize=(7,6))
    scatter = plt.scatter(
        Xp[:,0], Xp[:,1],
        c=c,
        cmap="tab10",
        alpha=0.6,
        s=15
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection colored by cluster")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.show()

    # Same PCA, colored by label
    plt.figure(figsize=(7,6))
    scatter = plt.scatter(
        Xp[:,0], Xp[:,1],
        c=y,
        cmap="tab10",
        alpha=0.6,
        s=15
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA projection colored by label")
    plt.colorbar(scatter, label="Label")
    plt.tight_layout()
    plt.show()


def routed_oof_macro_f1(
    X_in, y_in, feature_cols,
    cluster_scaler, clusterer,
    cluster_models, fallback_model,
    cv):
    X_in = X_in.reset_index(drop=True)
    y_in = y_in.reset_index(drop=True)

    y_oof = np.empty(len(X_in), dtype=int)

    # precompute cluster ids once (no leakage; clusterer already fixed)
    c_all = clusterer.predict(cluster_scaler.transform(X_in[feature_cols]))

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_in, y_in), 1):
        X_tr = X_in.iloc[tr_idx][feature_cols]
        y_tr = y_in.iloc[tr_idx]
        c_tr = c_all[tr_idx]

        X_te = X_in.iloc[te_idx][feature_cols]
        c_te = c_all[te_idx]

        y_hat = np.empty(len(te_idx), dtype=int)

        for k in np.unique(c_te):
            te_loc = np.where(c_te == k)[0]      # local indices in test fold
            Xk_te = X_te.iloc[te_loc]

            # train data restricted to cluster k
            tr_loc = np.where(c_tr == k)[0]
            if len(tr_loc) < 50:  # too small -> fallback
                m = clone(fallback_model)
                m.fit(X_tr, y_tr)
            else:
                model_k = cluster_models.get(int(k), None)
                if model_k is None:
                    m = clone(fallback_model)
                    m.fit(X_tr, y_tr)
                else:
                    m = clone(model_k)
                    m.fit(X_tr.iloc[tr_loc], y_tr.iloc[tr_loc])

            y_hat[te_loc] = m.predict(Xk_te).astype(int)

        y_oof[te_idx] = y_hat

    macroF1 = f1_score(y_in, y_oof, average="macro")
    return macroF1, y_oof


def tune_and_select_best_model(X, y, cv, random_state=42):
    """
    Tune a set of candidate models on (X, y) and return (best_name, best_estimator, best_cv_f1).
    X must already be the feature matrix for this cluster.

    ====================
    Model selection summary (CV macro-F1):
    ET: 0.745563
    CatBoost: 0.717845
    RF: 0.717530
    XGB: 0.711076
    HGB: 0.697853
    ====================

    """

    results = []
    Xtr = X
    ytr = y

    #ET
    best_et = tune_ET(Xtr, ytr, cv=cv)
    res = evaluate_experiment(
        name="ET",
        model=best_et,
        X=Xtr,
        y=ytr,
        cv=cv
    )

    print("Tuning ET:", res)
    f1_et = cross_val_score(best_et, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("ET", best_et, f1_et))

    # RF_V2
    best_rf = tune_random_forest(Xtr, ytr, cv=cv, n_iter=30, random_state=random_state)
    res_rf = evaluate_experiment(
        name="RF",
        model=best_rf,
        X=Xtr,
        y=ytr,
        cv=cv
    )
    print("Tuning RF:", res_rf)
    f1_rf = cross_val_score(best_rf, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("RF", best_rf, f1_rf))



    #uncomment to have the tuning of all the models!
    """ 
    # RF
    best_rf = tune_random_forest(Xtr, ytr, cv=cv, n_iter=30, random_state=random_state)
    f1_rf = cross_val_score(best_rf, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("RF", best_rf, f1_rf))
    
    #ET PCA
    best_et_PCA = tune_ET_with_PCA(Xtr, ytr, cv=cv)
    f1_et_PCA = cross_val_score(best_et, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("ET", best_et_PCA, f1_et_PCA))

    # CatBoost
    best_cb = tune_catboost(Xtr, ytr, cv=cv, n_iter=20, random_state=random_state)
    f1_cb = cross_val_score(best_cb, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("CatBoost", best_cb, f1_cb))

    # XGB
    n_classes = int(pd.Series(ytr).nunique())
    best_xgb = tune_xgb(Xtr, ytr, cv=cv, n_classes=n_classes, n_iter=20, random_state=random_state)
    f1_xgb = cross_val_score(best_xgb, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("XGB", best_xgb, f1_xgb))

  
    #LDA
    best_LDA =tune_LDA_shrinkage(Xtr, ytr, cv, n_iter=30, random_state=42)
    f1_LDA = cross_val_score(best_LDA, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("LDA", best_LDA, f1_LDA))


    # LGBM
    best_lgbm = tune_lgbm(Xtr, ytr, cv=cv, n_iter=10, random_state=random_state, class_weight="balanced")
    f1_lgbm = cross_val_score(best_lgbm, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("LGBM", best_lgbm, f1_lgbm))


    # HGB
    best_hgb = tune_hgb(Xtr, ytr, cv=cv, n_iter=20, random_state=random_state)
    f1_hgb = cross_val_score(best_hgb, Xtr, ytr, scoring="f1_macro", cv=cv, n_jobs=-1).mean()
    results.append(("HGB", best_hgb, f1_hgb))

    """

    best_name, best_model, best_f1 = max(results, key=lambda t: t[2])

    print("\nModel selection summary (cluster):")
    for name, _, score in sorted(results, key=lambda t: t[2], reverse=True):
        print(f"{name:>8s}: {score:.6f}")
    print(f"Selected: {best_name} with CV macro-F1={best_f1:.6f}")

    return best_name, best_model, best_f1



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

    experiments.append(("GMM_6comp", Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(
            n_components=6,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state
        ))
    ])))

    experiments.append(("GMM_4comp", Pipeline([
        ("scaler", StandardScaler()),
        ("gmm", GaussianMixture(
            n_components=4,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=random_state
        ))
    ])))
    
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


    result = {
        "experiment": name,
        "accuracy": acc,
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    # Per-class precision / recall / f1
    for cls in sorted(y.unique()):
        cls_str = str(cls)
        result[f"precision_{cls}"] = report[cls_str]["precision"]
        result[f"recall_{cls}"] = report[cls_str]["recall"]
        result[f"f1_{cls}"] = report[cls_str]["f1-score"]

    # Per-class accuracy
    acc_pc = per_class_binary_accuracy(y, y_pred)
    for cls, a in acc_pc.items():
        result[f"accuracy_{cls}"] = a

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
    counts = Counter(y)  # e.g. {0:..., 1:..., 2:..., 3:...}
    K_c = len(counts)
    N_c = len(y)
    w3 = {c: N_c / (K_c * counts[c]) for c in counts}  # "balanced" weights
    #extra customized weights
    w2 = {
        0: 1.1,
        1: 0.8,
        2: 1.1,
        3: 1.6
    }
    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 8, 12, 15, 20],
        "min_samples_split": [2, 4, 8, 12],
        "min_samples_leaf": [1, 2, 4, 6, 8, 16, 32],
        "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5, 0.8],
        "class_weight": [None, "balanced", "balanced_subsample", w3, w2]
    }

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


def tune_random_forest_v2(X, y, cv, n_iter=80, random_state=42):
    # balanced-ish custom weights (often helps macro-F1)
    counts = Counter(y)
    K = len(counts)
    N = len(y)
    w_bal = {c: N / (K * counts[c]) for c in counts}

    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": randint(400, 1400),
        "max_depth": [None, 10, 12, 15, 18, 22, 26, 30],
        "min_samples_split": randint(2, 16),
        "min_samples_leaf": randint(1, 12),
        "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5, 0.7],
        "bootstrap": [True, False],
        "class_weight": [None, "balanced", "balanced_subsample", w_bal],
    }

    # Note: max_samples only valid if bootstrap=True (sklearn handles invalid combos badly)
    # We'll do it by post-filtering: simplest is to omit it, OR run a second pass after.
    # (I recommend a second pass; see below.)

    rs = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    t0 = time.time()
    rs.fit(X, y)
    print(f"RF RandomizedSearchCV done in {time.time()-t0:.1f}s")
    print("Best RF params:", rs.best_params_)
    print("Best RF CV macro-F1:", rs.best_score_)

    return rs.best_estimator_, rs

def tune_LDA_shrinkage(X, y, cv, n_iter=30, random_state=42):
    """
    Tune LDA with shrinkage (the thing that helps for elongated / correlated clusters).
    We wrap it in a Pipeline with StandardScaler, then RandomizedSearchCV.

    Notes:
    - shrinkage is only supported with solver='lsqr' or 'eigen'
    - shrinkage='auto' is very strong; we also search numeric shrinkage in [0, 1]
    - LDA has no class_weight parameter. If class imbalance is severe, consider
      sample_weight in .fit(), but RandomizedSearchCV doesn’t handle that cleanly
      unless you pass fit_params. Start simple first.
    """

    lda_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
    ])

    # Search over solvers that support shrinkage
    # shrinkage can be 'auto' or a float in [0, 1]
    param_dist = [
        {
            "lda__solver": ["lsqr"],
            "lda__shrinkage": ["auto"] + list(np.linspace(0.0, 1.0, 21)),
        },
        {
            "lda__solver": ["eigen"],
            "lda__shrinkage": ["auto"] + list(np.linspace(0.0, 1.0, 21)),
        }
    ]

    rs = RandomizedSearchCV(
        lda_pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )

    t0 = time.time()
    rs.fit(X, y)
    print(f"RandomizedSearchCV done in {time.time()-t0:.0f}s")
    print("Best LDA params:", rs.best_params_)
    print("Best LDA CV macro-F1:", rs.best_score_)

    return rs.best_estimator_


def tune_ET_cluster(X, y, cv, random_state=42, n_iter=60):
    et = ExtraTreesClassifier(random_state=random_state, n_jobs=1)

    param_dist = {
        "n_estimators": randint(400, 1400),
        "max_depth": [None, 10, 12, 15, 20],
        "min_samples_split": randint(2, 12),
        # IMPORTANT: bias towards small leaves (macro-F1 often needs it)
        "min_samples_leaf": [1, 1, 1, 2, 2, 4, 8, 16],
        "max_features": ["sqrt", "log2", 0.1, 0.2, 0.3, 0.5, 0.8],
        "bootstrap": [False, True],
        "class_weight": ["balanced", "balanced_subsample", None],
    }

    rs = RandomizedSearchCV(
        et,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    rs.fit(X, y)

    print("Best ET params:", rs.best_params_)
    print("Best ET CV macro-F1:", rs.best_score_)
    return rs.best_estimator_

def tune_ET(X, y, cv):
    counts = Counter(y)  # e.g. {0:..., 1:..., 2:..., 3:...}
    K_c = len(counts)
    N_c = len(y)
    w3 = {c: N_c / (K_c * counts[c]) for c in counts}  # "balanced" weights

    et =  ExtraTreesClassifier(random_state=42, n_jobs=1)  # n_jobs=1 to avoid nested parallelism

    #extra customized weights
    w2 = {
        0: 1.1,
        1: 0.8,
        2: 1.1,
        3: 1.6
    }

    param_dist_et = {
        "n_estimators": [400, 600, 800, 1000],
        "max_depth": [None, 12, 15],
        "min_samples_split": [2, 4, 8],
        "min_samples_leaf": [1, 2, 4, 6, 8, 16, 32],
        "max_features": ["sqrt", 0.2, 0.3, 0.5, 0.8],
        "class_weight": ["balanced", "balanced_subsample", None, w3, w2]
    }
    
    rs_et = RandomizedSearchCV(
        et, param_distributions=param_dist_et, n_iter=30,
        scoring="f1_macro", cv=cv, random_state=42, n_jobs=-1, verbose=1
    )
    t0 = time.time()
    rs_et.fit(X, y)
    print(f"RandomizedSearchCV done in {time.time()-t0:.0f}s")
    best_et = rs_et.best_estimator_
    print("Best ET params:", rs_et.best_params_)
    print("Best ET CV macro-F1:", rs_et.best_score_)
    return rs_et.best_estimator_

def tune_xgb(X, y, cv, n_classes, n_iter=60, random_state=42):
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    param_dist = {
        "n_estimators": randint(300, 1400),
        "learning_rate": loguniform(1e-2, 2e-1),
        "max_depth": randint(3, 10),
        "min_child_weight": loguniform(1e-2, 20.0),
        "subsample": uniform(0.6, 0.4),  # 0.6..1.0
        "colsample_bytree": uniform(0.6, 0.4),  # 0.6..1.0
        "gamma": loguniform(1e-8, 1.0),
        "reg_alpha": loguniform(1e-8, 1e-1),
        "reg_lambda": loguniform(1e-2, 10.0),
    }

    rs = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"XGB RandomizedSearchCV done in {time.time() - t0:.0f}s")
    print("Best XGB params:", rs.best_params_)
    print("Best XGB CV f1_macro:", rs.best_score_)
    return rs.best_estimator_

def tune_hgb(X, y, cv, n_iter=60, random_state=42):
    hgb = HistGradientBoostingClassifier(random_state=random_state)
    param_dist = {
        "learning_rate": loguniform(1e-2, 2e-1),
        "max_iter": randint(200, 1200),
        "max_depth": randint(2, 10),
        "min_samples_leaf": randint(10, 120),
        "l2_regularization": loguniform(1e-6, 1e-1),
        "max_bins": randint(128, 256),
    }

    rs = RandomizedSearchCV(
        hgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"HGB RandomizedSearchCV done in {time.time()-t0:.0f}s")
    print("Best HGB params:", rs.best_params_)
    print("Best HGB CV f1_macro:", rs.best_score_)
    return rs.best_estimator_

def tune_lgbm(X, y, cv, n_iter=10, random_state=42, class_weight="balanced"):
    lgbm = LGBMClassifier(
        objective="multiclass",
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight
    )

    param_dist = {
        "n_estimators": randint(300, 2000),
        "learning_rate": loguniform(1e-2, 2e-1),
        "num_leaves": randint(16, 256),
        "max_depth": randint(-1, 16),  # -1 means no limit
        "min_child_samples": randint(10, 150),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "reg_alpha": loguniform(1e-8, 1e-1),
        "reg_lambda": loguniform(1e-2, 10.0),
    }

    rs = RandomizedSearchCV(
        lgbm,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"LGBM RandomizedSearchCV done in {time.time()-t0:.0f}s")
    print("Best LGBM params:", rs.best_params_)
    print("Best LGBM CV f1_macro:", rs.best_score_)
    return rs.best_estimator_

def tune_catboost(X, y, cv, n_iter=40, random_state=42):
    cb = CatBoostClassifier(
        loss_function="MultiClass",
        random_seed=random_state,
        verbose=False
    )
    param_dist = {
        "iterations": randint(400, 2500),
        "learning_rate": loguniform(1e-2, 2e-1),
        "depth": randint(4, 10),
        "l2_leaf_reg": loguniform(1.0, 50.0),
        "random_strength": loguniform(1e-3, 10.0),
        "bagging_temperature": uniform(0.0, 1.0),
    }

    rs = RandomizedSearchCV(
        cb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    t0 = time.time()
    rs.fit(X, y)
    print(f"CatBoost RandomizedSearchCV done in {time.time()-t0:.0f}s")
    print("Best CatBoost params:", rs.best_params_)
    print("Best CatBoost CV f1_macro:", rs.best_score_)
    return rs.best_estimator_
"""
def per_class_binary_accuracy(y_true, y_pred):
    
    #Per-class accuracy in a one-vs-rest (binary) sense:
    #Acc_i = (TP_i + TN_i) / N
    
    acc = {}
    classes = np.unique(y_true)
    N = len(y_true)

    for c in classes:
        y_true_bin = (y_true == c)
        y_pred_bin = (y_pred == c)

        TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))

        acc[c] = (TP + TN) / N

    return acc
"""

def per_class_binary_accuracy(y_true, y_pred):
    """
    One-vs-rest per-class accuracy:
    Acc_i = (TP_i + TN_i) / N

    Works safely with pandas Series or numpy arrays.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = {}
    classes = np.unique(y_true)
    N = y_true.shape[0]

    for c in classes:
        y_true_bin = (y_true == c)
        y_pred_bin = (y_pred == c)

        TP = np.sum(y_true_bin & y_pred_bin)
        TN = np.sum((~y_true_bin) & (~y_pred_bin))

        acc[c] = (TP + TN) / N

    return acc


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
    global CLUSTER_SCALER, CLUSTERER, OUTLIER_MODELS, OUTLIER_THRESHOLDS, CLASSIFIERS, TAU_BY_CLUSTER, LABEL_FP

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
    """
    print("\nPlotting PCA: clusters vs labels (INLIERS)...")
    plot_pca_cluster_vs_label(
        X,
        y,
        CLUSTER_SCALER,
        CLUSTERER,
        FEATURE_COLUMNS
    )

    """
    # after fitting clusterer on X_tr
    c_tr = CLUSTERER.predict(CLUSTER_SCALER.transform(X))
    sil = silhouette_score(CLUSTER_SCALER.transform(X), c_tr)
    print("silhouette:", sil)

    # distribution of clusters
    print("cluster sizes:", np.bincount(c_tr))


    #############################################
    # (a) Exploratory Data Analysis [6 points]  #
    #############################################
    #Please, uncomment to visualize plots.
    #make_EDA(X_train_full, y_train_full, X_out_cal)


    ##########################################
    #  (b)   Baseline Model [3 points]       #
    ##########################################

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

    # Fit once to report chosen hyperparameters (informational)
    grid.fit(X_train_full, y_train_full)
    print("Chosen K (baseline):", grid.best_params_["knn__n_neighbors"])
    print("Best inner-CV macro-F1 (baseline):", grid.best_score_)

    # Nested CV predictions (proper evaluation after tuning)
    y_pred_cv = cross_val_predict(grid, X_train_full, y_train_full, cv=cv, n_jobs=-1)

    acc = accuracy_score(y_train_full, y_pred_cv)
    print(f"\nBaseline KNN Accuracy (nested CV): {acc:.4f}\n")

    # Per-class precision/recall/F1 + macro avg + weighted avg
    report = classification_report(
        y_train_full, y_pred_cv,
        digits=4,
        output_dict=True
    )

    # Print per-class metrics
    print("Per-class Precision / Recall / F1:")
    classes = sorted(np.unique(y_train_full))
    for c in classes:
        c = str(c)
        print(
            f"  class {c}: "
            f"P={report[c]['precision']:.4f}  "
            f"R={report[c]['recall']:.4f}  "
            f"F1={report[c]['f1-score']:.4f}  "
            f"(n={int(report[c]['support'])})"
        )

    # Compute per-class accuracy
    acc_per_class = per_class_binary_accuracy(y_train_full, y_pred_cv)

    print("\nPer-class Accuracy (one-vs-rest):")
    for c, a in acc_per_class.items():
        print(f"  class {c}: accuracy = {a:.4f}")

    # Macro-averaged F1 (explicit)
    print(f"\nMacro-averaged F1: {report['macro avg']['f1-score']:.4f}")

    # (optional but nice) also show weighted F1
    print(f"Weighted-averaged F1: {report['weighted avg']['f1-score']:.4f}")

    #############################################################
    #  (c) Model experimentation and validation [19 points]     #
    #############################################################

    # Run candidate all experiments (on training set)
    """
    experiments = build_experiments(random_state=42)
    results_full = run_experiments(experiments, X_train_full, y_train_full, cv, save_path="experiment_results_full.csv", tag="FULL_")
    print("\nTop FULL results:")
    print(results_full[["experiment", "accuracy", "macro_f1"]].head(10))
    print("Finished testing all experiments on all dataset")
    

    #Train Random Forest with tuning all all the dataset
    #print("\nTuning RandomForest X_train_full...")
    #best_rf_in = tune_random_forest(X_train_full, y_train_full, cv=cv, n_iter=40, random_state=42)
    #FINAL_PIPELINE = Pipeline([("clf", best_rf_in)])
    """
    print("\nTuning models on D (mix inlier and outlier)...")
    _, best_rf_in, _ = tune_and_select_best_model(X_train_full, y_train_full, cv=cv, random_state=42)

    #FINAL_PIPELINE = best_rf_in
    FINAL_PIPELINE = Pipeline([("clf", best_rf_in)])
    y_pred_cv = cross_val_predict(
        FINAL_PIPELINE,
        X_train_full,
        y_train_full,
        cv=cv,
        n_jobs=-1
    )

    # Compute confusion matrix
    cm = confusion_matrix(y_train_full, y_pred_cv)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(np.unique(y_train_full)),
        yticklabels=sorted(np.unique(y_train_full))
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Final Pipeline, CV on inliers + outliers)")
    plt.tight_layout()
    plt.show()


    acc = accuracy_score(y_train_full, y_pred_cv)
    print(f"\nRF Accuracy (CV, on D): {acc:.4f}\n")

    print("RF Classification Report (CV, on D):")
    print(classification_report(y_train_full, y_pred_cv))

    #FINAL_PIPELINE.fit(X_train_full, y_train_full)

    scores = cross_val_score(FINAL_PIPELINE, X_train_full, y_train_full, scoring="f1_macro", cv=cv)
    f1_macro_mean = scores.mean()
    print("CV macro-F1: %.3f ± %.3f" % (scores.mean(), scores.std()))
    print("Finished tuning on all dataset")



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
    #plt.show()
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
    #plt.show()
    plt.close()


    X_in = X_train_full.loc[~is_out_train].reset_index(drop=True)
    y_in = y_train_full.loc[~is_out_train].reset_index(drop=True)
    #refit clusterer on X_in, then re-assign clusters and retrain models.
    #CLUSTER_SCALER, CLUSTERER, _ = fit_clusterer(X_in[FEATURE_COLUMNS], n_clusters=3, random_state=42)

    #print("\nPlotting PCA: clusters vs labels (INLIERS)...")
    #plot_pca_cluster_vs_label(
    #    X_in,
    #    y_in,
    #    CLUSTER_SCALER,
    #    CLUSTERER,
    #    FEATURE_COLUMNS
    #)


    #############################################################
    #  (e) Leaderboard Predictions [+ Bonus points]             #
    ############################################################

    experiments = build_experiments(random_state=42)
    results_full = run_experiments(experiments, X_in[FEATURE_COLUMNS], y_in, cv, save_path="experiment_results_inliers.csv", tag="INLIERS_")
    print("\nTop INLIERS results:")
    print(results_full[["experiment", "accuracy", "macro_f1"]].head(10))
    print("Finished testing all experiments on ONLY INLIERS dataset")

    """

    print("\nTuning ET on Din (inliers only)...")
    best_et_in = tune_ET(X_in[FEATURE_COLUMNS], y_in, cv=cv)
    print("Finished tuning ET on ONLY INLIERS dataset")
    print("\nValidating GLOBAL ET on Din (OOF)...")
    y_pred_global = cross_val_predict(
        best_et_in,
        X_in[FEATURE_COLUMNS],
        y_in,
        cv=cv,
        n_jobs=-1
    )
    print("GLOBAL ET report (OOF):")
    print(classification_report(y_in, y_pred_global, digits=4))

    """
    # Train a fallback model on ALL inliers (used if a cluster model is missing)
    #print("\nTraining fallback model on ALL inliers...")
    #_, fallback_clf, _ = tune_and_select_best_model(X_in[FEATURE_COLUMNS], y_in, cv=cv, random_state=42)
    #FALLBACK_MODEL = Pipeline([("clf", fallback_clf)])

    # Assign clusters for inliers
    Xs_in = CLUSTER_SCALER.transform(X_in[FEATURE_COLUMNS])
    c_in = CLUSTERER.predict(Xs_in)

    CLUSTER_MODELS = {}
    for k in sorted(np.unique(c_in)):
        idx = np.where(c_in == k)[0]
        Xk = X_in.iloc[idx][FEATURE_COLUMNS]
        yk = y_in.iloc[idx]

        print(f"\n=== Tuning models for cluster {k} | n={len(idx)} ===")
        # If cluster too small, skip and rely on fallback
        if len(idx) < 200:
            print(f"Cluster {k} too small (n={len(idx)}). Using fallback only.")
            continue
        """
        if k == 0:
            experiments = build_experiments(random_state=42)
            results_full = run_experiments(experiments, Xk, yk, cv,
                                           save_path=f"experiment_results_in_cluster_{k}.csv", tag=f"IN_CL_{k}_")
            print("\nTop Results CLUSTER {k} results:")
            print(results_full[["experiment", "accuracy", "macro_f1"]].head(10))
            print("Finished testing for CLUSTER {k}")
        print("tuning")
        """
        _, best_clf_k, _ = tune_and_select_best_model(Xk, yk, cv=cv, random_state=42)
        CLUSTER_MODELS[int(k)] = Pipeline([("clf", best_clf_k)])

    # Optional: if you want to keep FINAL_PIPELINE for compatibility/logging
    print("\nCluster-specific models trained:", sorted(CLUSTER_MODELS.keys()))

    # assign cluster ids for inliers
    c_in = CLUSTERER.predict(CLUSTER_SCALER.transform(X_in[FEATURE_COLUMNS]))

    for k in sorted(CLUSTER_MODELS.keys()):
        idx = np.where(c_in == k)[0]
        Xk = X_in.iloc[idx][FEATURE_COLUMNS]
        yk = y_in.iloc[idx]

        y_pred_k = cross_val_predict(CLUSTER_MODELS[k], Xk, yk, cv=cv, n_jobs=-1)

        print(f"\n=== Cluster {k} ===")
        print("n samples:", len(idx))
        print("macro-F1:", f1_score(yk, y_pred_k, average="macro"))
        print(classification_report(yk, y_pred_k, digits=4))

    print("Finished -by cluster- training")

    # -----------------------------
    # Train a fallback classifier on ALL inliers (safety net)
    # -----------------------------
    print("\nTraining FALLBACK_MODEL on all inliers...")
    _, best_fallback_clf, _ = tune_and_select_best_model(X_in[FEATURE_COLUMNS], y_in, cv=cv, random_state=42) #tune_ET(X_in[FEATURE_COLUMNS], y_in, cv=cv)  # or tune_and_select_best_model(...)
    y_pred_global = cross_val_predict(
        best_fallback_clf,
        X_in[FEATURE_COLUMNS],
        y_in,
        cv=cv,
        n_jobs=-1
    )
    # Compute confusion matrix
    cm = confusion_matrix(y_in, y_pred_global)

    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted(np.unique(y_in)),
        yticklabels=sorted(np.unique(y_in))
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Fallback Model, CV on Inliers)")
    plt.tight_layout()
    plt.show()


    #print("\n clusters trained:", sorted(CLUSTER_MODELS.keys()))
    # Build a fallback model and set FINAL_PIPELINE so predict()/submission checks don't fail
    FALLBACK_MODEL = Pipeline([("clf", clone(best_fallback_clf))])
    FALLBACK_MODEL.fit(X_in[FEATURE_COLUMNS], y_in)

    FINAL_PIPELINE = FALLBACK_MODEL  # compatibility/logging; routing uses CLUSTER_MODELS



    #SCORE 1 MODEL for all clusters
    scores = cross_val_score(FINAL_PIPELINE, X_in[FEATURE_COLUMNS], y_in, scoring="f1_macro", cv=cv)
    f1_macro_FINAL_PIPELINE = scores.mean()
    print("CV macro-F1: %.3f ± %.3f" % (scores.mean(), scores.std()))
    print("Finished tuning RF on all dataset")

    #SCORE 1 MODEL for EACH cluster
    f1_macro_CLUSTERS_MODELS, y_oof = routed_oof_macro_f1(
        X_in, y_in, FEATURE_COLUMNS,
        CLUSTER_SCALER, CLUSTERER,
        CLUSTER_MODELS, FALLBACK_MODEL,
        cv
    )
    print("Global OOF macro-F1 (class score):", f1_macro_CLUSTERS_MODELS)

    if f1_macro_FINAL_PIPELINE > f1_macro_CLUSTERS_MODELS:
        LABEL_FP = True
    else:
        LABEL_FP = False

    # -----------------------------
    # Finalize globals for predict()
    # -----------------------------
    OUTLIER_DETECTOR = DENSITY_MODEL
    THRESHOLD = TAU


    # Safety check (fail early with a useful message)
    if FINAL_PIPELINE is None:
        raise RuntimeError("FINAL_PIPELINE is None right before submission generation.")
    if OUTLIER_DETECTOR is None:
        raise RuntimeError("OUTLIER_DETECTOR is None right before submission generation.")
    if THRESHOLD is None:
        raise RuntimeError("THRESHOLD is None right before submission generation.")
    if FEATURE_COLUMNS is None:
        raise RuntimeError("FEATURE_COLUMNS is None right before submission generation.")

    print("Starting evalutation of score f1 macro outlier detections")
    print("Starting evalutation of score f1 macro outlier detections")
    f1_bin, f1_macro = proxy_outlier_f1_from_confident_inliers(
        OUTLIER_DETECTOR,
        TAU,
        X_train_full[FEATURE_COLUMNS],
        X_out[FEATURE_COLUMNS],
        q_conf=0.90
    )
    print("Proxy Outlier F1 (binary) detector outliers:", f1_bin)
    print("Proxy Outlier F1 (macro) detector outliers:", f1_macro)


    # compute synthetic leaderboard Score (adjust weights if assignment specifies)
    ClassScore_FINAL_PIPELINE = f1_macro_FINAL_PIPELINE * 100.0
    ClassScore_CLUSTERS_MODELS = f1_macro_CLUSTERS_MODELS * 100.0
    OutlierScore = f1_macro * 100.0
    CombinedScore_FINAL_PIPELINE = 0.8 * ClassScore_FINAL_PIPELINE + 0.2 * OutlierScore
    CombinedScore_CLUSTERS_MODELS = 0.8 * ClassScore_CLUSTERS_MODELS + 0.2 * OutlierScore

    print(f"\nSimulated Leaderboard Scores using FINAL PIPELINE -> Score: {CombinedScore_FINAL_PIPELINE:.2f}, ClassScore: {ClassScore_FINAL_PIPELINE:.2f}, OutlierScore: {OutlierScore:.2f}")
    print(f"\nSimulated Leaderboard Scores using CLUSTERS MODELS -> Score: {CombinedScore_CLUSTERS_MODELS:.2f}, ClassScore: {ClassScore_CLUSTERS_MODELS:.2f}, OutlierScore: {OutlierScore:.2f}")


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
