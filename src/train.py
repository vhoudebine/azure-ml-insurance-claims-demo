import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.2)
    parser.add_argument("--n_estimators", required=False, default=100, type=int)
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float)
    parser.add_argument("--registered_model_name", type=str, help="model name")

    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
   
    # Start Logging
    mlflow.start_run()

    #df = pd.read_csv(args.data,  storage_options = {'sas_token' : 'sp=r&st=2024-01-17T14:53:01Z&se=2024-01-31T22:53:01Z&spr=https&sv=2022-11-02&sr=c&sig=UgmbwOdnNLODC9whMO%2Fo5yofOsxf7Jgt96RdxCy%2BAYQ%3D'})
    df = pd.read_parquet(args.data)
    df['timestamp'] = pd.to_datetime(df.policy_created_date)
    df.sort_values(by='timestamp', inplace=True)

    # feature_eng
    for i in [30,60,90]:
        df[f'model_claims_sum_{i}_continuous'] = df.groupby(['make','model'])[['is_claim','timestamp']]\
                                    .rolling(f'{i}D', on='timestamp',closed='neither')\
                                    .sum()[['is_claim','timestamp']]\
                                    .reset_index(level=['make','model'])['is_claim']

    train_df = df.copy()
    train_df.drop(['policy_id','policy_created_date'], axis=1, inplace=True)
    train_df = train_df[train_df['timestamp']<pd.to_datetime('now')]
    train_df.set_index('timestamp', inplace=True)
    train_df.sort_index(inplace=True)

    X = train_df.drop(labels=['is_claim'], axis=1)
    y = train_df['is_claim']

    test_size = args.test_train_ratio

    ## Time aware cross validation split to preserve order
    n_splits = (1//test_size)-1   # using // for integer division

    tscv = TimeSeriesSplit(n_splits=int(n_splits))
    for train_index, test_index in tscv.split(X):
        print(train_index, test_index)

    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    num_cols = X_train.select_dtypes(exclude=["object"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

    num_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

    cat_pipe = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="N/A"), OneHotEncoder(handle_unknown="ignore", sparse=False)
    )

    full_pipe = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    pipeline = make_pipeline(full_pipe, 
                        GradientBoostingClassifier(n_estimators=100, 
                                                    learning_rate=0.1,
                                                    verbose=True))

    # Model training + evaluation
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    roc_auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1], multi_class='ovr')
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", report['weighted avg']['precision'])
    mlflow.log_metric("recall", report['weighted avg']['recall'])

    print("Registering the model via MLFlow")
    signature = infer_signature(X_train)
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        registered_model_name=args.registered_model_name,
        artifact_path='model',
        signature=signature,
    )


    mlflow.end_run()

if __name__ == "__main__":
    main()