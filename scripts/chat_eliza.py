from src.chatting import eliza
from src.paths import LOCAL_RAW_DATA_PATH

eliza = eliza.Eliza()
eliza.load(LOCAL_RAW_DATA_PATH / 'eliza/doctor.txt')

print(eliza.initial())
while True:
    said = input('> ')
    response = eliza.respond(said)
    if response is None:
        break
    print(response)
print(eliza.final())