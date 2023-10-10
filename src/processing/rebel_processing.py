from transformers import pipeline

# Function to parse the generated text and extract the triplets
def extract_triplets(text):
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({'subject': subject.strip(), 'relation': relation.strip(),'object': object_.strip()})
    return triplets

def run_inference_and_extract_triplets(model, batched_input_text):
    predicted_batch_labels = []

    outputs = model(batched_input_text, return_tensors=True, return_text=False)
    
    for output in outputs:
        generated_ids = output["generated_token_ids"]
        extracted_text = model.tokenizer.batch_decode([generated_ids.cpu()])  # Move to CPU for decoding
        extracted_triplets = extract_triplets(extracted_text[0])
        predicted_batch_labels.append(extracted_triplets)

    return predicted_batch_labels



def get_model(device, base_model='Babelscape/rebel-large'):
    model = pipeline('text2text-generation', model=base_model, tokenizer=base_model, device=device)
    return model