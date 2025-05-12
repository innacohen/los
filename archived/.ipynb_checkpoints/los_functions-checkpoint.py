import pandas as pd
import numpy as np
import os
from datetime import datetime

#ML packages
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.discretisation import DecisionTreeDiscretiser
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score, classification_report, precision_recall_curve, roc_curve, auc
from sklearn.calibration import calibration_curve

#XGBo
#https://github.com/liannewriting/YouTube-videos-public/blob/main/xgboost-python-tutorial-example/xgboost_python.ipynb
import xgboost as xgb
from category_encoders.target_encoder import TargetEncoder
from xgboost import XGBClassifier
#hyperparameter values
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

#Feature Selection
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection
from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropFeatures, DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection
from feature_engine.imputation import AddMissingIndicator
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.imputation import CategoricalImputer
from feature_engine.encoding import RareLabelEncoder

#Plotting Packages
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from hyperopt import tpe
from hyperopt import fmin, tpe, Trials
from hyperopt import STATUS_OK
from hyperopt import hp
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


def map_med (value):
    if value in ["Cardiology", "Medicine"]:
        return "Medicine"
    else:
        return value

def preprocess_features(feature_fp, df_filt):
    
    features_los = pd.read_excel(feature_fp)

    # Features that are divided into 5 types to conduct data preprocessing
    # 1. drop: features that need to be dropped
    drop_list = features_los[features_los['type'] == 'drop']['col_name'].tolist()

    # 2. category: features that are already categorical and need to be OneHotEncoding (add missing value indicator, impute missing by adding 'missing' category)
    category_list = features_los[features_los['type'] == 'category']['col_name'].tolist()

    # 3. binary: features that are binary and need to be converted to categorical (add missing value indicator / prefer: fill missing with 0)
    binary_list = features_los[features_los['type'] == 'binary']['col_name'].tolist()

    # 4. continuous: features that are continous/numerical variables, need outlier handling and normalization (add missing value indicator, fill missing with median)
    continuous_list = features_los[features_los['type'] == 'continuous']['col_name'].tolist()

    # 5. discrete: features that are discrete/numerical variables, need to be discretized (fill missing with 0)
    discrete_list = features_los[features_los['type'] == 'discrete']['col_name'].tolist()

    # Split into X, y
    X = df_filt.drop(['viz_outcome_prolonged_los_yn'], axis=1)
    y = df_filt['viz_outcome_prolonged_los_yn']

    # Convert variables to categorical
    X[category_list] = X[category_list].astype('category')
    
    #Drop variables reset index
    continuous_list.remove("viz_age")
    X = X.drop(columns=drop_list+continuous_list)
    X = X.reset_index()
    y = y.reset_index()

    # Split the data by group shuffle split on 'PAT_MRN_ID' into train set and validation set
    gss = GroupShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    
    train_ix, val_ix = next(gss.split(X, y, groups=X['pat_mrn_id']))
    
    X_train = X.loc[train_ix]
    y_train = y.loc[train_ix]
    
    X_val = X.loc[val_ix]
    y_val = y.loc[val_ix]
    
    # Drop 'PAT_MRN_ID' and set 'PAT_ENC_CSN_ID' as index
    X_train = X_train.drop(['pat_mrn_id'], axis=1).set_index('pat_enc_csn_id')
    X_val = X_val.drop(['pat_mrn_id'], axis=1).set_index('pat_enc_csn_id')
    y_train = y_train.set_index('pat_enc_csn_id')
    y_val = y_val.set_index('pat_enc_csn_id')

        # Update feature preprocessing pipeline
    feature_preprocess_pipeline = Pipeline(steps=[
    
        # Missing value imputation
        # Impute missing values with 0 for discrete variables
        ('arbitrary_number_imputer', ArbitraryNumberImputer(arbitrary_number=0, variables=discrete_list)),
    
        #Update 08/01/2024. Replaced CategoricalImputer with ArbitraryNumberInputer for Binary variables (1/0)
        # Impute missing values with 0 for binary variables 
        ('binary_imputer', ArbitraryNumberImputer(arbitrary_number=0, variables=binary_list)),
        
        # Impute missing values with adding 'missing' category for categorical variables 
       ('categorical_imputer', CategoricalImputer(variables=category_list)),
        
        # Rare encoding for categorical variables
        ('rare_label_encoder', RareLabelEncoder(tol=0.01, n_categories=5, max_n_categories=10, variables=category_list)),
    
        # OneHotEncoding for categoricals
        ('one_hot_category', OneHotEncoder(variables=category_list)),
        
    ])  # Apply the pipeline

    # Apply the pipeline
    X_train_preprocessed = feature_preprocess_pipeline.fit_transform(X_train, y_train)
    X_val_preprocessed = feature_preprocess_pipeline.transform(X_val)

    print(f"df_filt shape: {df_filt.shape}")
    print(f"X train_preprocessed shape: {X_train_preprocessed.shape}")
    print(f"X val preprocessed shape: {X_val_preprocessed.shape}")
    print(f"y train prolonged LOS proportion: {y_train['viz_outcome_prolonged_los_yn'].sum()/y_train.shape[0]}")
    print(f"y val prolonged LOS proportion: {y_val['viz_outcome_prolonged_los_yn'].sum()/y_val.shape[0]}")

    
    return (X_train_preprocessed, X_val_preprocessed, y_train, y_val)


def select_features(X_train_preprocessed, X_val_preprocessed, y_train):
        # 1st Feature Selection pipeline
    feature_selection_pipeline = Pipeline(steps=[
    
            ('drop_constant', DropConstantFeatures(tol=0.99)),
    
            ('drop_duplicates', DropDuplicateFeatures()),
    
            ('correlated_features', SmartCorrelatedSelection(
            method='pearson',
            threshold=0.9,
            selection_method='model_performance',
            estimator=xgb.XGBClassifier(random_state=0)
            ))
    ])

    # Apply the pipeline
    X_train_selected = feature_selection_pipeline.fit_transform(X_train_preprocessed, y_train)
    X_val_selected = feature_selection_pipeline.transform(X_val_preprocessed)

    print(f"X_train_selected shape: {X_train_selected.shape}")
    print(f"X_val_selected shape: {X_val_selected.shape}")


    return(X_train_selected, X_val_selected)

# Choose best hyperparameters
def objective_function_xgb_1b(params, X_train, y_train):
    clf = xgb.XGBClassifier(**params)
    auc_scorer = make_scorer(roc_auc_score)
    score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', error_score='raise').mean()
    return {'loss': -score, 'status': STATUS_OK}

space_xgb= {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
    'max_depth': hp.choice('max_depth', np.arange(5, 15+1, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(5, 35+1, dtype=int)),
    'num_leaves': hp.choice('num_leaves', np.arange(5, 50+1, dtype=int)),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
}

def run_xgb_tune_fit_1b(X_train, y_train, num_eval):
    trials = Trials()
    best_params = fmin(fn=lambda params: objective_function_xgb_1b(params, X_train, y_train),
                       space=space_xgb, algo=tpe.suggest, verbose=4, max_evals=num_eval, trials=trials,
                       rstate= np.random.default_rng(0))

    best_params['max_depth'] = list(np.arange(5, 15+1, dtype=int))[best_params['max_depth']]
    best_params['n_estimators'] = list(np.arange(5, 35+1, dtype=int))[best_params['n_estimators']]
    best_params['num_leaves'] = list(np.arange(5, 50+1, dtype=int))[best_params['num_leaves']]
    
    best_xgb_model = xgb.XGBClassifier(**best_params)
    print(f"XGB Best parameters for {len(X_train.columns)} features: {best_params}")
    best_xgb_model.fit(X_train, y_train)
    return best_xgb_model

# Evaluate model performance including auc, accuracy, etc
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    pr_auc_score = auc(recall, precision)

    return {
        'y_test': y_test,
        'predictions': predictions,
        'probabilities': probabilities,
        'auc_score': auc_score,
        'accuracy': accuracy,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'pr_auc_score': pr_auc_score
    }

def plot_metrics(data):
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))  # 3 rows x 1 column

    # Plot ROC Curve
    y_test = data['y_test']
    y_probs_positive = data['probabilities']
    fpr, tpr, _ = roc_curve(y_test, y_probs_positive)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()

    # Plot PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_probs_positive)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, label=f'PR-AUC = {pr_auc:.2f}')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('PR Curve')
    axes[1].legend()

    # Plot Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_probs_positive, n_bins=10)
    axes[2].plot(prob_pred, prob_true, label='Calibration')
    axes[2].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title('Calibration Curve')
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    return fig 


def plot_cm(data):
    y_test = data['y_test']['viz_outcome_prolonged_los_yn']
    y_pred = data['predictions']
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    
    # Extract values for binary classification
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    
    # Calculate metrics
    top_row = tn + fp
    bot_row = fn + tp
    
    specificity_value = recall_score(y_test, y_pred, pos_label=0)
    sensitivity_value = recall_score(y_test, y_pred, pos_label=1)
    accuracy_value = accuracy_score(y_test, y_pred)
    precision_value = precision_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred)
    
    # Print results
    print(f'Specificity : {specificity_value:.4f}')
    print(f'Sensitivity : {sensitivity_value:.4f}')
    print(f'Accuracy : {accuracy_value:.4f}')
    print(f'Precision : {precision_value:.4f}')
    print(f'F1 score : {f1_value:.4f}')
    
    print(f'Of {top_row} people who did not have a prolonged LOS, {tn} ({specificity_value:.2%}) were correctly classified.')
    print(f'Of {bot_row} people who did have a prolonged LOS, {tp} ({sensitivity_value:.2%}) were correctly classified.')
    
    return fig

def plot_shap(model_xgb, X_val_selected):
    np.random.seed(42)
    explainer = shap.Explainer(model_xgb)
    shap_values = explainer(X_val_selected)
    shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
    mean_abs_shap = np.abs(shap_df).mean(axis=0)
    top_30_features = mean_abs_shap.sort_values(ascending=False).head(30)
    
    for feature in top_30_features.index.tolist():
        print(feature)

    lab_df = pd.read_csv('/gpfs/milgram/project/rtaylor/imc33/LOS/data/new_label_names.csv')
    
    new_feature_names_dict = pd.Series(lab_df.new_name.values, index=lab_df.old_name).to_dict()
    
    current_feature_names = shap_values.feature_names  # Current feature names
    
    # Update feature names in the SHAP values object
    new_feature_names = [new_feature_names_dict.get(name, name) for name in current_feature_names]
    shap_values.feature_names = new_feature_names

    # Create a figure for the beeswarm plot
    fig = plt.figure(figsize=(10, 8))
    
    # Plot the SHAP values
    shap.plots.beeswarm(shap_values, max_display=30)
    
    # Return the figure
    return fig



def run_xgb_and_plot(df_filt, fp, output_folder, df_name, suffix):
    X_train_preprocessed, X_val_preprocessed, y_train, y_val = preprocess_features(fp, df_filt)
    X_train_selected, X_val_selected = select_features(X_train_preprocessed, X_val_preprocessed, y_train)
    model_xgb = run_xgb_tune_fit_1b(X_train_selected, y_train, num_eval=20)
    results = evaluate_model(model_xgb, X_val_selected, y_val)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the current date
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Save the plots to the output folder with df_name, suffix, and date in the filenames
    metrics_plot_path = os.path.join(output_folder, f'metrics_plot_{df_name}_{suffix}_{date_str}.png')
    cm_plot_path = os.path.join(output_folder, f'confusion_matrix_{df_name}_{suffix}_{date_str}.png')
    shap_plot_path = os.path.join(output_folder, f'shap_plot_{df_name}_{suffix}_{date_str}.png')

    fig = plot_metrics(results)
    fig.savefig(metrics_plot_path)
    plt.close(fig)

    fig = plot_cm(results)
    fig.savefig(cm_plot_path)
    plt.close(fig)

    fig = plot_shap(model_xgb, X_val_selected)
    fig.savefig(shap_plot_path)
    plt.close(fig)

    print(f"Plots for {df_name}_{suffix}_{date_str} saved to {output_folder}")



def run_xgb_and_plot(df_filt, fp, output_folder, df_name, suffix):
    X_train_preprocessed, X_val_preprocessed, y_train, y_val = preprocess_features(fp, df_filt)
    X_train_selected, X_val_selected = select_features(X_train_preprocessed, X_val_preprocessed, y_train)
    model_xgb = run_xgb_tune_fit_1b(X_train_selected, y_train, num_eval=20)
    results = evaluate_model(model_xgb, X_val_selected, y_val)

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the current date
    date_str = datetime.now().strftime('%Y-%m-%d')

    # Save the plots to the output folder with df_name, suffix, and date in the filenames
    metrics_plot_path = os.path.join(output_folder, f'metrics_plot_{df_name}_{suffix}_{date_str}.png')
    cm_plot_path = os.path.join(output_folder, f'confusion_matrix_{df_name}_{suffix}_{date_str}.png')
    shap_plot_path = os.path.join(output_folder, f'shap_plot_{df_name}_{suffix}_{date_str}.png')

    fig = plot_metrics(results)
    fig.savefig(metrics_plot_path)
    plt.close(fig)

    fig = plot_cm(results)
    fig.savefig(cm_plot_path)
    plt.close(fig)

    fig = plot_shap(model_xgb, X_val_selected)
    fig.savefig(shap_plot_path)
    plt.close(fig)

    print(f"Plots for {df_name}_{suffix}_{date_str} saved to {output_folder}")