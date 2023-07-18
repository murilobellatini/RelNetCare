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
from src.paths import LOCAL_PROCESSED_DATA_PATH
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


def load_and_preprocess_data(data_dir):
    
    spacy_entity_map = {
        "PER": "PERSON",
        "STRING": "PRODUCT",  # Approximating common nouns to PRODUCT, @todo: use NOUN strategy.
        "GPE": "GPE",
        "VALUE": "QUANTITY",
        "ORG": "ORG",
    }

    dt = DialogREDatasetTransformer(LOCAL_PROCESSED_DATA_PATH / data_dir)
    df = dt.load_data_to_dataframe()

    df_relations = df.explode('Relations').apply(lambda r: {**{"Origin": r['Origin'], 'Dialogue': r['Dialogue']}, **r['Relations']}, axis=1)
    df_relations = pd.json_normalize(df_relations)

    mask = df_relations.min_words_distance.isna()
    df_relations = df_relations.dropna()
    df_relations['r'] = df_relations['r'].str[0]

    df_relations['x_type'] = df_relations['x_type'].map(spacy_entity_map)
    df_relations['y_type'] = df_relations['y_type'].map(spacy_entity_map)
    df_relations = mark_entities(df_relations)

    return df_relations

def feature_engineering(df_relations):

    le_dict = {}  # Create a dictionary to store LabelEncoders

    for col in ['r', 'x_type', 'y_type',
                 'spacy_features.x_pos', 'spacy_features.x_dep',
                 'spacy_features.x_tag', 'spacy_features.y_pos',
                 'spacy_features.y_dep', 'spacy_features.y_tag'
                ]:
        le = LabelEncoder()
        df_relations[col] = le.fit_transform(df_relations[col])
        le_dict[col] = le  # Store the fitted LabelEncoder in the dictionary

    scaler = None
    # scaler = StandardScaler()

    # scaled_data = scaler.fit_transform(df_relations['min_words_distance'].values.reshape(-1, 1))
    # df_relations['min_words_distance_scaled'] = scaled_data
    # df_relations['min_turn_distance_scaled'] = scaler.fit_transform(df_relations['min_turn_distance'].values.reshape(-1, 1))

    # Extract token span start and end positions from 'x_token_span' and 'y_token_span' columns
    # df_relations['x_token_span_start'] = df_relations.x_token_span.apply(lambda x: x[0])
    # df_relations['x_token_span_end'] = df_relations.x_token_span.apply(lambda x: x[1])
    # df_relations['y_token_span_start'] = df_relations.y_token_span.apply(lambda x: x[0])
    # df_relations['y_token_span_end'] = df_relations.y_token_span.apply(lambda x: x[1])

    add_dialogue_as_features = True
    vectorizer = None
    if add_dialogue_as_features:
        vectorizer = TfidfVectorizer(
            stop_words=stop_words
            )
        TFIDF = vectorizer.fit_transform(df_relations['Dialogue'].apply(lambda x: ' '.join(x))).toarray()
        tfidf_df = pd.DataFrame(TFIDF, columns=vectorizer.get_feature_names_out())
        df_relations = pd.concat([df_relations.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    train_data = df_relations[(df_relations['Origin'] == 'train') | (df_relations['Origin'] == 'dev')]
    test_data = df_relations[df_relations['Origin'] == 'test']
    dev_data = df_relations[df_relations['Origin'] == 'dev']

    drop_cols = ['x','y','r', 't', 'rid', 
                 'Origin', 'Dialogue', 
                 'x_token_span', 'y_token_span',
                 'x_char_span', 'y_char_span',
                 #'min_words_distance', 
                 'min_words_distance_pct',
                 #'min_turn_distance',
                 'min_turn_distance_pct', 
                #  'spacy_features.x_pos', 'spacy_features.x_dep',
                #  'spacy_features.x_tag', 'spacy_features.y_pos',
                #  'spacy_features.y_dep', 'spacy_features.y_tag'
                 ]

    drop_cols = [col for col in drop_cols if col in train_data.columns]

    X_train = train_data.drop(drop_cols, axis=1)
    X_test = test_data.drop(drop_cols, axis=1)
    X_dev = dev_data.drop(drop_cols, axis=1)

    y_train = train_data['r']
    y_test = test_data['r']
    y_dev = dev_data['r']

    return X_train, X_test, X_dev, y_train, y_test, y_dev, vectorizer, le_dict, scaler


def train_model(X_train, X_test, X_dev, y_train, y_test, y_dev, epoch_cnt, patience=None):
    D_train = xgb.DMatrix(X_train, label=y_train)
    D_test = xgb.DMatrix(X_test, label=y_test)
    D_dev = xgb.DMatrix(X_dev, label=y_dev)

    xgb_params = {
        'eta': 0.5,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': y_train.max() + 1
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
        
        best_preds_train = np.asarray([np.argmax(line) for line in preds_train])
        best_preds_test = np.asarray([np.argmax(line) for line in preds_test])

        f1_train = f1_score(y_train, best_preds_train, average='binary')
        f1_test = f1_score(y_test, best_preds_test, average='binary')

        results = {
            'loss': evals_result['train']['mlogloss'][i],
            'eval_loss': evals_result['eval']['mlogloss'][i],
            'f1': f1_test,
            'epoch': i,
        }
        
        print('f1_train=', f1_train)
        print('f1_test=', f1_test)

        wandb.log(results)

    run.finish()

    return model

def evaluate_model(model, X_test, y_test, X_dev, y_dev):
    D_test = xgb.DMatrix(X_test)

    preds_test = model.predict(D_test)

    best_preds_test = np.asarray([np.argmax(line) for line in preds_test])

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
    for le_key, le in le_dict.items():
        pickle.dump(le, open(os.path.join(path, f'label_encoder_{le_key}.pkl'), 'wb'))
    pickle.dump(vectorizer, open(os.path.join(path, 'vectorizer.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(path, 'scaler.pkl'), 'wb'))

def load_model(path):
    model = pickle.load(open(os.path.join(path, 'model.pkl'), 'rb'))
    le = pickle.load(open(os.path.join(path, 'label_encoder.pkl'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(path, 'vectorizer.pkl'), 'rb'))
    scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))
    return model, le, vectorizer, scaler


if __name__ == "__main__":
    patience= None
    add_dialogue_as_features = True
    epoch_cnt = 20
    data_dir = 'dialog-re-binary-enriched'
    df_relations = load_and_preprocess_data(data_dir)
    X_train, X_test, X_dev, y_train, y_test, y_dev, vectorizer, le_dict, scaler = feature_engineering(df_relations)
    model = train_model(X_train, X_test, X_dev, y_train, y_test, y_dev, epoch_cnt, patience)
    evaluate_model(model, X_test, y_test, X_dev, y_dev)
    save_model(model, le_dict, vectorizer, scaler, path)
    model, le_dict, vectorizer, scaler = load_model(path)
