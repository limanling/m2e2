from collections import defaultdict

def event_type_norm(type_str):
    return type_str.replace('.', '||').replace(':', '||').replace('-', '|').upper()


def role_name_norm(type_str):
    return type_str.upper()

entity_type_mapping_brat = {
    'PER': 'PER',
    'ORG': 'ORG',
    'GPE': 'GPE',
    'LOC': 'LOC',
    'FAC': 'FAC',
    'VEH': 'VEH',
    'WEA': 'WEA',
    # 'TIM': 'TIME',
    # 'NUM': 'VALUE',
    # 'TIT',
    'MON': 'VALUE',
    # 'URL',
    # 'RES',
    # 'BALLOT'
}

event_type_mapping_brat2ace = {
    'Die': 'Life:Die',
    # 'Injure': 'Life:Injure',
    'TransferMoney': 'Transaction:Transfer-Money',
    'Attack': 'Conflict:Attack',
    'Demonstrate': 'Conflict:Demonstrate',
    'Correspondence': 'Contact:Phone-Write',
    'Meet': 'Contact:Meet',
    'ArrestJail': 'Justice:Arrest-Jail',
    # 'ReleaseParole': 'Justice:Release-Parole',
    'TransportPerson': 'Movement:Transport',
    'TransportArtifact': 'Movement:Transport',
}

event_type_mapping_ace2brat = {event_type_norm(v): k for k, v in event_type_mapping_brat2ace.items()}
# print(event_type_mapping_ace2brat)

event_type_mapping_aida2brat = {
    'Life.Die': 'Die',
    'Life.Injure': 'Injure',
    'Transaction.TransferMoney': 'TransferMoney',
    'Conflict.Attack': 'Attack',
    'Conflict.Demonstrate': 'Demonstrate',
    'Contact.Correspondence': 'Correspondence',
    'Contact.Meet': 'Meet',
    'Justice.ArrestJail': 'ArrestJail',
    'Justice.ReleaseParole': 'ReleaseParole',
    'Movement.TransportPerson': 'TransportPerson',
    'Movement.TransportArtifact': 'TransportArtifact',
}

event_role_mapping_brat2ace = { # delete time-related ones when testing
    'Die': {'Victim': 'Victim', 'Agent': 'Agent', 'Instrument': 'Instrument', 'Place': 'Place'},  #, 'Time': 'Time'},
    # 'Injure': {'Victim': 'Victim', 'Agent': 'Agent', 'Instrument': 'Instrument', 'Place': 'Place'},  #, 'Time': 'Time'},
    'TransferMoney': {'Giver': 'Giver', 'Recipient': 'Recipient', 'Beneficiary': 'Beneficiary', 'Money': 'Money', 'Place': 'Place'},  #, 'Time': 'Time'},
    'Attack': {'Attacker': 'Attacker', 'Instrument': 'Instrument', 'Place': 'Place', 'Target': 'Target'},  #, 'Time': 'Time'},
    'Demonstrate': {'Demonstrator': 'Entity', 'Place': 'Place'},  #, 'Time': 'Time'},
    'Correspondence': {'Participant': 'Entity', 'Place': 'Place'},  #, 'Time': 'Time'},
    'Meet': {'Participant': 'Entity', 'Place': 'Place'},  #, 'Time': 'Time'},
    'ArrestJail': {'Agent': 'Agent', 'Person': 'Person', 'Place': 'Place'},  #, 'Time': 'Time'},
    # 'ReleaseParole': {'Agent': 'Entity', 'Person': 'Person', 'Place': 'Place'},  #, 'Time': 'Time'},
    'TransportPerson': {'Agent': 'Agent', 'Person': 'Artifact', 'Instrument': 'Vehicle', 'Destination': 'Destination', 'Origin': 'Origin'},  #, 'Time': 'Time'},
    'TransportArtifact': {'Agent': 'Agent', 'Artifact': 'Artifact', 'Instrument': 'Vehicle', 'Destination': 'Destination', 'Origin': 'Origin'},  #, 'Time': 'Time'},
}

# event_role_mapping_ace2brat = {role_name_norm(v): k for t in event_role_mapping_brat2ace.items() for k, v in event_role_mapping_brat2ace[t]}
event_role_mapping_ace2brat = defaultdict(lambda : defaultdict())
for t in event_role_mapping_brat2ace:
    for k, v in event_role_mapping_brat2ace[t].items():
        event_type_ace = event_type_norm(event_type_mapping_brat2ace[t])
        event_role_mapping_ace2brat[event_type_ace][role_name_norm(v)] = k
# print(event_role_mapping_ace2brat)

event_type_mapping_image2ace = {
    'Movement.TransportPerson': 'Movement:Transport',
    'Movement.TransportArtifact': 'Movement:Transport',
    'Life.Die': 'Life:Die',
    'Conflict.Attack': 'Conflict:Attack',
    'Conflict.Demonstrate': 'Conflict:Demonstrate',
    'Contact.Phone-Write': 'Contact:Phone-Write',
    'Contact.Meet': 'Contact:Meet',
    'Transaction.TransferMoney': 'Transaction:Transfer-Money',
    'Justice.ArrestJail': 'Justice:Arrest-Jail',
}

event_role_mapping_image2ace = {
    'Life.Die': {'victim': 'Victim', 'agent': 'Agent', 'instrument': 'Instrument', 'place': 'Place'},  #, 'Time': 'Time'},
    'Transaction.TransferMoney': {'instrument': 'Instrument', 'giver': 'Giver', 'recipient': 'Recipient', 'beneficiary': 'Beneficiary', 'money': 'Money', 'place': 'Place'},  #, 'Time': 'Time'},
    'Conflict.Attack': {'attacker': 'Attacker', 'instrument': 'Instrument', 'place': 'Place', 'target': 'Target', 'victim': 'Target'},  #, 'Time': 'Time'},
    'Conflict.Demonstrate': {'police':'Police', 'instrument': 'Instrument', 'demonstrator': 'Entity', 'place': 'Place', 'participant': 'Entity'},  #, 'Time': 'Time'},
    'Contact.Phone-Write': {'instrument':'Instrument', 'participant': 'Entity', 'place': 'Place'},  #, 'Time': 'Time'},
    'Contact.Meet': {'participant': 'Entity', 'place': 'Place'},  #, 'Time': 'Time'},
    'Justice.ArrestJail': {'instrument': 'Instrument', 'agent': 'Agent', 'person': 'Person', 'place': 'Place'},  #, 'Time': 'Time'},
    'Movement.TransportPerson': {'agent': 'Agent', 'person': 'Artifact', 'instrument': 'Instrument', 'destination': 'Destination', 'origin': 'Origin'},  #, 'Time': 'Time'},
    'Movement.TransportArtifact': {'person': 'Artifact', 'agent': 'Agent', 'artifact': 'Artifact', 'instrument': 'Instrument', 'destination': 'Destination', 'origin': 'Origin'},  #, 'Time': 'Time'},
}
# 'TransportArtifact' has 'person' in image annotation