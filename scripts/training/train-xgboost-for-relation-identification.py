import re
import os
import pickle
import wandb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.utils import to_camel_case
from src.paths import LOCAL_PROCESSED_DATA_PATH, LOCAL_MODELS_PATH
from src.processing.dialogre_processing import DialogREDatasetTransformer
from src.processing.dataframe_utils import get_counts_and_percentages

import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

# Get the English stopwords
stop_words = stopwords.words('english')

# set entity token markers
ENTITY_X_TOKEN = 'x_marker'
ENTITY_Y_TOKEN = 'y_marker'

def mark_entities(df_relations):
    df_relations['Dialogue'] = df_relations.apply(lambda row: 
                                                 [re.sub(r'\b' + re.escape(row['x']) + r'\b', ENTITY_X_TOKEN, 
                                                    re.sub(r'\b' + re.escape(row['y']) + r'\b', ENTITY_Y_TOKEN, sentence))
                                                 for sentence in row['Dialogue']],
                                                 axis=1)
    return df_relations


def load_and_preprocess_data(data_path):

    df = load_data(data_path)
    df_relations = preprocess_data(df=df)

    return df_relations

def load_data(data_path):

    dt = DialogREDatasetTransformer(data_path)
    df = dt.load_data_to_dataframe()
    
    return df

def preprocess_data(df, mode='train'):
        
    spacy_entity_map = {
        "PER": "PERSON",
        "STRING": "PRODUCT",  # Approximating common nouns to PRODUCT, @todo: use NOUN strategy.
        "GPE": "GPE",
        "VALUE": "QUANTITY",
        "ORG": "ORG",
    }
    
    df_relations = df.explode('Relations').apply(lambda r: {**{"Origin": r['Origin'], 'Dialogue': r['Dialogue']}, **r['Relations']}, axis=1)
    df_relations = pd.json_normalize(df_relations)

    mask = df_relations.min_words_distance.isna()
    df_relations = df_relations.dropna()

    if mode == 'train':
        df_relations['r'] = df_relations['r'].str[0]
        df_relations['x_type'] = df_relations['x_type'].map(spacy_entity_map)
        df_relations['y_type'] = df_relations['y_type'].map(spacy_entity_map)
    df_relations = mark_entities(df_relations)
    
    return df_relations


def feature_engineering(df_relations, mode='train', label_encoders=None, vectorizers=None):

    le_dict = {} if label_encoders is None else label_encoders
    for col in ['r', 'x_type', 'y_type']:
        if mode == 'train':
            le = LabelEncoder()
            df_relations[col] = le.fit_transform(df_relations[col])
            le_dict[col] = le
        else:
            df_relations[col] = le_dict[col].transform(df_relations[col])

    scaler = None
    add_dialogue_as_features = True
    vectorizer = vectorizers
    if add_dialogue_as_features:
        if mode == 'train':
            vectorizer = TfidfVectorizer(stop_words=stop_words)
            TFIDF = vectorizer.fit_transform(df_relations['Dialogue'].apply(lambda x: ' '.join(x))).toarray()
        else:
            TFIDF = vectorizer.transform(df_relations['Dialogue'].apply(lambda x: ' '.join(x))).toarray()

        tfidf_df = pd.DataFrame(TFIDF, columns=vectorizer.get_feature_names_out())
        df_relations = pd.concat([df_relations.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    train_data = df_relations[df_relations['Origin'] == 'train']
    test_data = df_relations[df_relations['Origin'] == 'test']
    dev_data = df_relations[df_relations['Origin'] == 'dev']

    drop_cols = ['r', 'x', 'y', 't', 'rid', 
                 'Origin', 'Dialogue', 
                 'x_token_span', 'y_token_span',
                 'x_char_span', 'y_char_span',
                 'min_words_distance_pct',
                 'min_turn_distance_pct', 
                 'spacy_features.x_pos', 'spacy_features.x_dep',
                 'spacy_features.x_tag', 'spacy_features.y_pos',
                 'spacy_features.y_dep', 'spacy_features.y_tag'
                 ]

    if mode == 'infer':
        drop_cols.append('r')

    drop_cols = [col for col in drop_cols if col in train_data.columns]

    X_train = train_data.drop(drop_cols, axis=1)
    X_test = test_data.drop(drop_cols, axis=1)
    X_dev = dev_data.drop(drop_cols, axis=1)

    y_train = train_data['r'] if mode == 'train' else None
    y_test = test_data['r'] if mode == 'train' else None
    y_dev = dev_data['r'] if mode == 'train' else None

    return X_train, X_test, X_dev, y_train, y_test, y_dev, vectorizer, le_dict, scaler


def train_model(X_train, X_test, X_dev, y_train, y_test, y_dev, epoch_cnt, patience=None):
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    D_dev = xgb.DMatrix(X_dev, label=y_dev)
    
    # count the instances of each class
    count_class_0, count_class_1 = np.bincount(y_train)
    # Use this count to calculate the scale_pos_weight value
    scale_pos_weight = count_class_0 / count_class_1

    xgb_params = {
        'eta': 0.5,
        'max_depth': 3,
        'objective': 'binary:logistic',  # changed from 'multi:softprob' to 'binary:logistic'
        # 'num_class': 1, # unncessary for binary classifier
        'scale_pos_weight': scale_pos_weight  # add scale_pos_weight to the parameters
    }

    run = wandb.init(reinit=True, project="RelNetCare", config=xgb_params)

    watchlist = [(D_train, 'train'), (D_dev, 'eval')]
    evals_result = {}
    if patience:
        model = xgb.train(xgb_params, D_train, num_boost_round=epoch_cnt, evals=watchlist, 
                        early_stopping_rounds=patience, evals_result=evals_result)
    else:
        model = xgb.train(xgb_params, D_train, num_boost_round=epoch_cnt,
                          evals=watchlist, evals_result=evals_result)
    for i in range(epoch_cnt):
        try:
            preds_train = model.predict(D_train, ntree_limit=i+1)
            preds_test = model.predict(D_test, ntree_limit=i+1)
        except xgb.core.XGBoostError:
            break
        
        if xgb_params['objective'] == 'binary:logistic':
            best_preds_train = np.where(preds_train > 0.5, 1, 0)
            best_preds_test = np.where(preds_test > 0.5, 1, 0)
            loss_key = 'logloss'
        else:
            best_preds_train = np.asarray([np.argmax(line) for line in preds_train])
            best_preds_test = np.asarray([np.argmax(line) for line in preds_test])
            loss_key = 'mlogloss'

        f1_train = f1_score(y_train, best_preds_train, average='binary')
        f1_test = f1_score(y_test, best_preds_test, average='binary')

        results = {
            'loss': evals_result['train'][loss_key][i],
            'eval_loss': evals_result['eval'][loss_key][i],
            'f1': f1_test,
            'epoch': i,
        }
        
        wandb.log(results)

    run.finish()

    return model

def evaluate_model(model, X_test, y_test, X_dev, y_dev):
    D_test = xgb.DMatrix(X_test)

    preds_test = model.predict(D_test)

    best_preds_test = np.where(preds_test > 0.5, 1, 0)

    print("Test Accuracy =", accuracy_score(y_test, best_preds_test))

    print("Test Classification Report:")
    print(classification_report(y_test, best_preds_test))

    feature_importance = model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Score'])
    feature_importance = feature_importance.sort_values(by='Score', ascending=False)
    print(feature_importance.head(20))

def save_model(model, le_dict, vectorizer, scaler, path):
    os.makedirs(path, exist_ok=True)
    pickle.dump(model, open(os.path.join(path, 'model.pkl'), 'wb'))
    pickle.dump(le_dict, open(os.path.join(path, 'label_encoder_dict.pkl'), 'wb'))
    pickle.dump(vectorizer, open(os.path.join(path, 'vectorizer.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(path, 'scaler.pkl'), 'wb'))

def load_model(path):
    model = pickle.load(open(os.path.join(path, 'model.pkl'), 'rb'))
    le_dict = pickle.load(open(os.path.join(path, 'label_encoder_dict.pkl'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(path, 'vectorizer.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
    return model, le_dict, vectorizer, scaler


if __name__ == "__main__":
    patience = 3
    add_dialogue_as_features = True
    epoch_cnt = 100
    data_dir = 'dialog-re-binary-validated-enriched'
    data_path = LOCAL_PROCESSED_DATA_PATH / data_dir
    model_path = LOCAL_MODELS_PATH / f'custom/relation-identification/xgboost/{data_dir}'
    df_relations = load_and_preprocess_data(data_path)
    X_train, X_test, X_dev, y_train, y_test, y_dev, vectorizer, le_dict, scaler = feature_engineering(df_relations)
    model = train_model(X_train, X_test, X_dev, y_train, y_test, y_dev, epoch_cnt, patience)
    evaluate_model(model, X_test, y_test, X_dev, y_dev)
    save_model(model, le_dict, vectorizer, scaler, model_path)
    model, le_dict, vectorizer, scaler = load_model(model_path)
