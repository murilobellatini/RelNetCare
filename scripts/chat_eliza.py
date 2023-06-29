import spacy
from spacy import displacy
from src.chatting import eliza
from src.paths import LOCAL_RAW_DATA_PATH

nlp = spacy.load('en_core_web_sm')

eliza = eliza.Eliza()
eliza.load(LOCAL_RAW_DATA_PATH / 'eliza/doctor.txt')

print(eliza.initial())

detected_entities = []
while True:
    print('detected_entities=',detected_entities)
    said = input('> ')
    doc = nlp(said)
    entities = [ent.text for ent in doc.ents]
    detected_entities.extend(entities)

    response = eliza.respond(said)
    if response is None:
        break
    print(response)

print(eliza.final())
