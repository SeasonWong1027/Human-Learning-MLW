import pandas as pd
import numpy as np
import optuna
import os
import json
import warnings
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

#This version was originally built for Windows, thus all output will be add "_win"

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

LOG_CSV = "V39_Checkpoint_mac.csv"
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", encoding='utf-8') as f:
        f.write("TRIAL,SCORE,WEIGHT,THRESHOLD,PARAMS\n")

def write_checkpoint(trial_num, score, weight, threshold, params):
    with open(LOG_CSV, "a", encoding='utf-8') as f:
        f.write(f"{trial_num},{score:.5f},{weight:.3f},{threshold:.3f},\"{str(params)}\"\n")
        f.flush()
        os.fsync(f.fileno())

def engineer_features(df):
    df = df.copy()
    df['Is_Infant'] = (df['Age'] <= 4).astype(int)
    df[['GROUP', 'ID']] = df['PassengerId'].str.split('_', expand=True)
    df['GroupSize'] = df['GROUP'].map(df['GROUP'].value_counts().to_dict())
    
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Num'] = pd.to_numeric(df['Num'])
    df['CabinRegion'] = pd.qcut(df['Num'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    
    fin_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in fin_cols:
        df.loc[df['CryoSleep'] == True, col] = 0
        df.loc[df['Age'] <= 12, col] = 0
    df['TotalSpend'] = df[fin_cols].sum(axis=1)
    
    for col in fin_cols + ['TotalSpend']:
        df[col] = np.log1p(df[col])
        
    drop_list = ['PassengerId', 'Cabin', 'GROUP', 'ID', 'Num', 'Name']
    df.drop(columns=[c for c in drop_list if c in df.columns], inplace=True)
    
    return df

if __name__ == '__main__':
    print("=== V39: CATBOOST + XGBOOST OPTUNA STACKING (Multi-Rank Mac Edition) ===")
    print("Hardware Target: MacOS | CPU (CatBoost) \n")
    
    TRAIN_DF = pd.read_csv('train.csv')
    TEST_DF = pd.read_csv('test.csv')
    TEST_PASSENGER_ID = TEST_DF['PassengerId'].copy()

    train_rows = TRAIN_DF.shape[0]
    Y = TRAIN_DF['Transported'].astype(int).values
    full_df = engineer_features(pd.concat([TRAIN_DF.drop(columns=['Transported']), TEST_DF]))

    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'CabinRegion']
    for col in cat_cols:
        full_df[col] = full_df[col].astype(str).replace('nan', 'Unknown')

    num_cols = full_df.select_dtypes(include=['number']).columns.tolist()
    imputer = SimpleImputer(strategy='median')
    full_df[num_cols] = imputer.fit_transform(full_df[num_cols])
    full_df = pd.get_dummies(full_df, columns=cat_cols)

    XY_TRAIN = full_df.iloc[:train_rows].values
    XY_TEST = full_df.iloc[train_rows:].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("[Phase 1] Building CatBoost Anchor (V34 Best Params) on CPU...")
    cb_params = {
        'depth': 4, 
        'iterations': 1000, 
        'l2_leaf_reg': 3.0, 
        'learning_rate': 0.03,
        'verbose': 0,
        'random_state': 42,
        'thread_count': -1  
    }
    
    cb_oof_proba = np.zeros(len(Y))
    for train_idx, val_idx in skf.split(XY_TRAIN, Y):
        X_tr, X_va = XY_TRAIN[train_idx], XY_TRAIN[val_idx]
        y_tr, y_va = Y[train_idx], Y[val_idx]
        
        cb_model = CatBoostClassifier(**cb_params)
        cb_model.fit(X_tr, y_tr, verbose=0)
        cb_oof_proba[val_idx] = cb_model.predict_proba(X_va)[:, 1]
    
    print("-> Training CatBoost on full data for final inference...")
    cb_full_model = CatBoostClassifier(**cb_params)
    cb_full_model.fit(XY_TRAIN, Y, verbose=0)
    cb_test_proba = cb_full_model.predict_proba(XY_TEST)[:, 1]
    
    print("CatBoost Anchor ready. VRAM released for XGBoost.\n")

    def objective(trial):
        xgb_params = {
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 6]),
            'n_estimators': trial.suggest_categorical('n_estimators', [300, 500, 1000, 1200, 1500]),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'tree_method': 'hist',
            'device': 'cuda', 
            'random_state': 42,
            'verbosity': 0
        }
        
        weight = trial.suggest_float('weight', 0.0, 1.0)
        threshold = trial.suggest_float('threshold', 0.35, 0.65)
        
        xgb_oof_proba = np.zeros(len(Y))
        
        for train_idx, val_idx in skf.split(XY_TRAIN, Y):
            X_tr, X_va = XY_TRAIN[train_idx], XY_TRAIN[val_idx]
            y_tr, y_va = Y[train_idx], Y[val_idx]
            
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            xgb_oof_proba[val_idx] = xgb_model.predict_proba(X_va)[:, 1]
            
        final_oof_proba = (weight * xgb_oof_proba) + ((1.0 - weight) * cb_oof_proba)
        final_preds = (final_oof_proba >= threshold).astype(int)
        cv_score = accuracy_score(Y, final_preds)
        
        write_checkpoint(trial.number, cv_score, weight, threshold, xgb_params)
        return cv_score

    print("[Phase 2] Initiating Optuna Stacking Optimization (XGBoost on RTX 4060)...")
    study_name = "V39_Stacking_mac"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        direction='maximize', 
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True
    )
    
    completed_trials = len(study.trials)
    target_trials = 200
    remaining_trials = max(0, target_trials - completed_trials)
    
    if completed_trials > 0:
        print(f"Resuming from DB: {completed_trials} completed. Best CV so far: {study.best_value:.5f}")
    
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)

    print(f"\nOptimization Complete! Global Best Stacking CV: {study.best_value:.5f}")

    print("\n[Phase 3] Extracting multi-rank parameters and generating submissions...")
    
    trials_df = study.trials_dataframe().sort_values(by='value', ascending=False).reset_index(drop=True)
    target_ranks = [1, 2, 3, 4, 5, 8, 10, 15, 24]
    
    with open('V39_parameter_mac.txt', 'w', encoding='utf-8') as f:
        f.write("=== V39 Multi-Rank Output Parameters ===\n\n")
        f.write(f"Global Best CV: {study.best_value:.5f}\n\n")
        
        for rank in target_ranks:
            if rank <= len(trials_df):
                target_row = trials_df.iloc[rank - 1]
                cv_score = target_row['value']
                weight = target_row['params_weight']
                threshold = target_row['params_threshold']
                
                xgb_params = {
                    'max_depth': int(target_row['params_max_depth']),
                    'n_estimators': int(target_row['params_n_estimators']),
                    'learning_rate': target_row['params_learning_rate'],
                    'subsample': target_row['params_subsample'],
                    'colsample_bytree': target_row['params_colsample_bytree'],
                    'tree_method': 'hist',
                    'device': 'cuda',
                    'random_state': 42,
                    'verbosity': 0
                }
                
                print(f"\n---> Processing Rank {rank} (CV: {cv_score:.5f}) <---")
                print(f"Fusion ratio: XGBoost ({weight*100:.1f}%) + CatBoost ({(1-weight)*100:.1f}%), Threshold: {threshold:.3f}")
                
                f.write(f"--- Rank {rank} ---\n")
                f.write(f"CV Score: {cv_score:.5f}\n")
                f.write(f"Fusion Configuration: XGBoost Weight {weight:.4f} | CatBoost Weight {(1.0 - weight):.4f}\n")
                f.write(f"Decision Threshold: {threshold:.4f}\n")
                f.write(f"XGBoost Parameters:\n{json.dumps(xgb_params, indent=4)}\n\n")

                xgb_model = xgb.XGBClassifier(**xgb_params)
                xgb_model.fit(XY_TRAIN, Y)
                xgb_test_proba = xgb_model.predict_proba(XY_TEST)[:, 1]
                
                final_test_proba = (weight * xgb_test_proba) + ((1.0 - weight) * cb_test_proba)
                final_preds = (final_test_proba >= threshold).astype(bool)
                
                filename = f'V39_submission_rank{rank}_mac.csv'
                submission = pd.DataFrame({
                    'PassengerId': TEST_PASSENGER_ID, 
                    'Transported': final_preds
                })
                submission.to_csv(filename, index=False)
                print(f"File saved successfully: {filename}")

    print("\nAll target rank submissions and parameter logs generated successfully.")