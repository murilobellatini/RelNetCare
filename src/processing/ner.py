import spacy
from spacy import displacy
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


class EntityProcessor:
    def __init__(self, df:pd.DataFrame, spacy_model="en_core_web_lg") -> None:
        self.nlp = spacy.load(spacy_model)
        self.df = df
        self.docs = None
        self.enriched_df = None
        
    def process_all_documents(self):
        all_dialogues = self.df.Dialogue.apply(lambda x: '\n'.join(x))
        self.docs = list(tqdm(self.nlp.pipe(all_dialogues), total=len(all_dialogues)))
        return self.docs

    def find_missing_entities(self, row):
        y_true = set(row['StandardizedAnnotatedEntities'])
        y_pred = set(row['PredictedEntities'])

        missing_from_ground_truth = list(y_pred.difference(y_true))
        missing_from_predictions = list(y_true.difference(y_pred))

        return pd.Series({'MissingFromGroundTruth': missing_from_ground_truth,
                          'MissingFromPredictions': missing_from_predictions})

    def find_correct_predictions(self, row):
        y_true = set(row['StandardizedAnnotatedEntities'])
        y_pred = set(row['PredictedEntities'])

        correct_predictions = list(y_true.intersection(y_pred))

        return pd.Series({'CorrectPredictions': correct_predictions})

    def calculate_classification_metrics(self, row):
        y_true = set(row['StandardizedAnnotatedEntities'])
        y_pred = set(row['PredictedEntities'])

        # We're treating this as a binary classification problem. Entity is either correct (1) or not (0).
        y_true_bin = [1 if entity in y_true else 0 for entity in y_true.union(y_pred)]
        y_pred_bin = [1 if entity in y_pred else 0 for entity in y_true.union(y_pred)]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, average='binary', zero_division=1)

        return pd.Series({'Precision': precision, 'Recall': recall, 'F1': f1})

    def extract_unique_entities_relations(self, row_relations, ignore_containing=['Speaker ']):
        # Initialize a set to store the unique entities
        unique_entities = set()

        # Iterate over each relation in the row
        for relation in row_relations:
            # Add the 'x' and 'y' entities and their types to the set
            if not any(sub_str in relation['x'] for sub_str in ignore_containing):
                unique_entities.add(f"{relation['x']}:{relation['x_type']}")
            if not any(sub_str in relation['y'] for sub_str in ignore_containing):
                unique_entities.add(f"{relation['y']}:{relation['y_type']}")

        return list(unique_entities)

    def extract_unique_entities_spacy(self, doc):
        unique_entities = set()
        
        entity = ""
        entity_type = ""
        
        for token in doc:
            if token.ent_iob_ == "B":
                # If an entity is currently being constructed, add it to the set
                if entity:
                    entity_processed = entity.strip()
                    if entity_processed.endswith("'s"):
                        entity_processed = entity_processed[:-2].strip()
                    unique_entities.add(f"{entity_processed}:{entity_type}")
                
                # Start a new entity
                entity = token.text
                entity_type = token.ent_type_
            elif token.ent_iob_ == "I":
                # Continue the entity
                entity += " " + token.text
            else:
                # If an entity is currently being constructed, add it to the set
                if entity:
                    entity_processed = entity.strip()
                    if entity_processed.endswith("'s"):
                        entity_processed = entity_processed[:-2].strip()
                    unique_entities.add(f"{entity_processed}:{entity_type}")
                # Reset the entity
                entity = ""
                entity_type = ""
        
        # If an entity is currently being constructed at the end of the document, add it to the set
        if entity:
            entity_processed = entity.strip()
            if entity_processed.endswith("'s"):
                entity_processed = entity_processed[:-2].strip()
            unique_entities.add(f"{entity_processed}:{entity_type}")
        
        return list(unique_entities)

    def standardize_entities(self, df: pd.DataFrame, type_mapping: dict = None) -> pd.DataFrame:
        """
        Standardize the entities in the 'AnnotatedEntities' column of the DataFrame
        by applying the type mapping. If no type_mapping is provided, a default
        mapping will be used.

        Args:
            df (pd.DataFrame): DataFrame containing the 'AnnotatedEntities' column.
            type_mapping (dict, optional): Mapping from annotation types to SpaCy types.
                Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with the standardized entities.
        """
        # Define the default type_mapping if none is provided
        if type_mapping is None:
            type_mapping = {
                'PER': 'PERSON',
                'STRING': 'STRING',
                'GPE': 'GPE',
                'ORG': 'ORG',
                'VALUE': 'CARDINAL'
            }
            
        # Apply the mapping to the 'AnnotatedEntities' column
        tmp_df = df['AnnotatedEntities'].apply(lambda entities:
            [f"{entity.split(':')[0]}:{type_mapping.get(entity.split(':')[1], 'OTHER')}" for entity in entities])

        return tmp_df

    def enrich_data(self, type_mapping: dict = None, ignore_substring_entities=["Speaker "]) -> pd.DataFrame:
        """
        Enrich the input DataFrame by applying all the available methods.

        Args:
            type_mapping (dict, optional): Mapping from annotation types to SpaCy types.
                Defaults to None.

        Returns:
            pd.DataFrame: Enriched DataFrame with all columns.
        """
    # Copy the original DataFrame to avoid modifying the original data
        
        enriched_df = self.df.copy()
        # Check if the documents have been processed
        
        if self.docs is None:
            self.docs = self.process_all_documents()

        # Extract predicted entities using spaCy for each document
        enriched_df['PredictedEntities'] = [self.extract_unique_entities_spacy(doc) for doc in self.docs]

        # Extract unique entities from relations column
        enriched_df['AnnotatedEntities'] = enriched_df['Relations'].apply(self.extract_unique_entities_relations, ignore_containing=ignore_substring_entities)

        # Standardize the unique entities using the provided type_mapping
        enriched_df['StandardizedAnnotatedEntities'] = self.standardize_entities(df=enriched_df, type_mapping=type_mapping)

        # Calculate classification metrics for each row in the DataFrame
        metrics_df = enriched_df.apply(self.calculate_classification_metrics, axis=1)

        # Join the metrics DataFrame with the original DataFrame
        enriched_df = enriched_df.join(metrics_df)

        # Find missing entities for each row
        enriched_df[['MissingFromGroundTruth', 'MissingFromPredictions']] = enriched_df.apply(self.find_missing_entities, axis=1)

        # Find correct predictions for each row
        enriched_df['CorrectPredictions'] = enriched_df.apply(self.find_correct_predictions, axis=1)

        # Return the enriched DataFrame
        self.enriched_df = enriched_df
        
        return self.enriched_df
    
    
    def validate_metrics(self, index, return_dialogue=True, show_precomputed=False):
        
        if self.enriched_df is None:
            self.enrich_data()
            
        row = self.enriched_df.iloc[index]

        TP = len(row.CorrectPredictions)
        FP = len(row.MissingFromGroundTruth)
        FN = len(row.MissingFromPredictions)

        # Calculate Precision
        if TP + FP > 0:
            Precision = TP / (TP + FP)
        else:
            Precision = 0.0

        # Calculate Recall
        if TP + FN > 0:
            Recall = TP / (TP + FN)
        else:
            Recall = 0.0

        # Calculate F1 Score
        if Precision + Recall > 0:
            F1 = 2 * (Precision * Recall) / (Precision + Recall)
        else:
            F1 = 0.0

        print(40*'=')
        print(f"# SAMPLE INSPECTION - #{index}")
        print(40*'=')

        if return_dialogue:
            displacy.render(self.docs[index], style="ent", jupyter=True)
            print(40*'-')
            
            
        print(f"## GT: MissingFromPredictions (FN={FN})")
        print(40*'-')
        print('\n'.join(row['MissingFromPredictions']))
        print(40*'-')

        print(f"## PREDICTIONS: MissingFromGroundTruth (FP={FP})")
        print(40*'-')
        print('\n'.join(row['MissingFromGroundTruth']))
        print(40*'-')
        
        print(f"## PREDICTIONS: CorrectPredictions (TP={TP})")
        print(40*'-')
        print('\n'.join(row['CorrectPredictions']))
        print(40*'-')
        
        print(40*'=')
        print("## VALIDATED METRICS")
        print(40*'=')
        if show_precomputed:
            print("### Pre-computed (df)")
            print(row[['Precision', 'Recall', 'F1']])

        print("### Post-computed (counts above)")

        print(f"""----------------------------------------
- Precision: {Precision:.1%} \tTP / (TP + FP) = {TP} / ({TP} + {FP})
- Recall:    {Recall:.1%} \tTP / (TP + FN) = {TP} / ({TP} + {FN})
- F1 Score:  {F1:.1%} \t2 * Pre * Rec / (Prec + Rec) = 2 * {Precision:.1%} * {Recall:.1%} / ({Precision:.1%} + {Recall:.1%})""")
        print(40*'=')
    
