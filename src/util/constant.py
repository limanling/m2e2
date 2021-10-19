"""
Define constants.
"""
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
# SUBJ is trigger, no type
# SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'Weapon': 2, 'Organization': 3, 'Geopolitical_Entity': 4, 'Location': 5, 'Time': 6, 'Vehicle': 7, 'Value': 8, 'Facility': 9, 'Person': 10}

NER_NORM = {'WEA': 'Weapon', 'ORG': 'Organization', 'GPE': 'Geopolitical_Entity', 'LOC': 'Location', 'TIME': 'Time', 'VEH': 'Vehicle', 'VALUE': 'Value', 'FAC': 'Facility', 'PER': 'Person'}

NER_NIL_LABEL = 'O'
NUM_OTHER_NER = 3
NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, NER_NIL_LABEL: 2, 'Weapon': 3, 'Organization': 4, 'Geopolitical_Entity': 5, 'Location': 6, 'Time': 7, 'Vehicle': 8, 'Value': 9, 'Facility': 10, 'Person': 11}
# {'ORG:Sports', 'VALUE:other', 'GPE:GPE-Cluster', 'VEH:Subarea-Vehicle', 'WEA:Nuclear', 'ORG:Religious', 'ORG:Government', 'FAC:Building-Grounds', 'LOC:Address', 'PER:Indeterminate', 'LOC:Boundary', 'ORG:Entertainment', 'WEA:Exploding', 'PER:Group', 'WEA:Blunt', 'VEH:Underspecified', 'WEA:Projectile', 'GPE:Nation', 'VEH:Water', 'GPE:Special', 'LOC:Region-General', 'LOC:Celestial', 'GPE:County-or-District', 'VEH:Air', 'WEA:Sharp', 'ORG:Educational', 'PER:Individual', 'FAC:Airport', 'WEA:Chemical', 'GPE:Population-Center', 'LOC:Region-International', 'TIME:other', 'ORG:Media', 'LOC:Water-Body', 'WEA:Biological', 'FAC:Subarea-Facility', 'WEA:Shooting', 'ORG:Commercial', 'ORG:Non-Governmental', 'GPE:Continent', 'ORG:Medical-Science', 'WEA:Underspecified', 'FAC:Path', 'GPE:State-or-Province', 'VEH:Land', 'LOC:Land-Region-Natural', 'FAC:Plant'

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}

DEPREL_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'punct': 2, 'compound': 3, 'case': 4, 'nmod': 5, 'det': 6, 'nsubj': 7, 'amod': 8, 'conj': 9, 'dobj': 10, 'ROOT': 11, 'cc': 12, 'nmod:poss': 13, 'mark': 14, 'advmod': 15, 'appos': 16, 'nummod': 17, 'dep': 18, 'ccomp': 19, 'aux': 20, 'advcl': 21, 'acl:relcl': 22, 'xcomp': 23, 'cop': 24, 'acl': 25, 'auxpass': 26, 'nsubjpass': 27, 'nmod:tmod': 28, 'neg': 29, 'compound:prt': 30, 'mwe': 31, 'parataxis': 32, 'root': 33, 'nmod:npmod': 34, 'expl': 35, 'csubj': 36, 'cc:preconj': 37, 'iobj': 38, 'det:predet': 39, 'discourse': 40, 'csubjpass': 41}

NEGATIVE_LABEL_TRIGGER = 'other_event'
NUM_OTHER_TRIGGER = 2
LABEL_TO_ID_TRIGGER = {PAD_TOKEN: 0, NEGATIVE_LABEL_TRIGGER: 1, 'Conflict:Attack': 2, 'Conflict:Demonstrate': 3,
                       'Contact:Meet': 4, 'Contact:Phone-Write': 5, 'Life:Die': 6, 'Movement:Transport': 7,
                       'Justice:Arrest-Jail': 8, 'Transaction:Transfer-Money': 9} # 'Life:Injure', 'Justice:Release-Parole'
LABEL_TO_ID_TRIGGER_UNSEEN = {'Personnel:Elect': 2,
                       'Life:Marry': 3, 'Life:Injure': 4, 'Justice:Execute': 5, 'Justice:Trial-Hearing': 6,
                       'Life:Be-Born': 7, 'Justice:Convict': 8, 'Justice:Release-Parole': 9, 'Justice:Fine': 10}
LABEL_TO_ID_TRIGGER_UNVISUAL = {'Personnel:End-Position': 3, 'Personnel:Start-Position': 4, 'Personnel:Nominate': 5,
                             'Justice:Sue': 7, 'Business:End-Org': 9, 'Business:Start-Org': 11,
                             'Transaction:Transfer-Ownership': 10, 'Justice:Sentence': 13,
                             'Justice:Charge-Indict': 16, 'Business:Declare-Bankruptcy': 18,
                             'Justice:Pardon': 21, 'Justice:Appeal': 22,  'Justice:Extradite': 23,
                             'Life:Divorce': 24, 'Business:Merge-Org': 25, 'Justice:Acquit': 26}
LABEL_TO_ID_TRIGGER_ALL = {PAD_TOKEN: 0, UNK_TOKEN: 1, NEGATIVE_LABEL_TRIGGER: 2, 'Personnel:Elect': 3,
                       'Personnel:End-Position': 4, 'Personnel:Start-Position': 5,
                       'Movement:Transport': 6, 'Conflict:Attack': 7, 'Personnel:Nominate': 8, 'Contact:Meet': 9,
                       'Life:Marry': 10,
                           'Justice:Sue': 11, 'Contact:Phone-Write': 12, 'Transaction:Transfer-Money': 13,
                       'Conflict:Demonstrate': 14, 'Life:Injure': 15, 'Business:End-Org': 16, 'Life:Die': 17,
                       'Justice:Arrest-Jail': 18, 'Transaction:Transfer-Ownership': 19, 'Business:Start-Org': 20,
                       'Justice:Execute': 21, 'Justice:Sentence': 22, 'Justice:Trial-Hearing': 23, 'Life:Be-Born': 24,
                       'Justice:Charge-Indict': 25, 'Justice:Convict': 26, 'Business:Declare-Bankruptcy': 27,
                       'Justice:Release-Parole': 28, 'Justice:Fine': 29, 'Justice:Pardon': 30, 'Justice:Appeal': 31,
                       'Justice:Extradite': 32, 'Life:Divorce': 33, 'Business:Merge-Org': 34, 'Justice:Acquit': 35}

NEGATIVE_LABEL_ROLE = 'other_role'
NUM_OTHER_ROLE = 2
LABEL_TO_ID_ROLE = {PAD_TOKEN: 0, NEGATIVE_LABEL_ROLE: 1, 'Buyer': 2, 'Target': 3, 'Agent': 4, 'Vehicle': 5,
                    'Instrument': 6, 'Person': 7, 'Victim': 8, 'Attacker': 9, 'Artifact': 10, 'Seller': 11,
                    'Recipient': 12, 'Money': 13, 'Giver': 14, 'Entity': 15, 'Place': 16, 'Defendant': 17,
                    'Destination': 18, 'Origin': 19}
LABEL_TO_ID_ROLE_UNSEEN = {'Beneficiary': 2, 'Prosecutor': 3,
                           'Plaintiff': 4, 'Adjudicator': 5, 'Agent': 6, 'Vehicle': 7}
LABEL_TO_ID_ROLE_UNVISUAL = {'Time-Holds': 8, 'Time-Starting': 10, 'Price': 11, 'Time-Before': 12,
                             'Time-After': 14, 'Time-Ending': 19, 'Time-At-End': 22,
                             'Time-At-Beginning': 28, 'Time-Within': 32, 'Org': 18,
                             'Sentence': 15, 'Crime': 16, 'Position': 24 }  #
LABEL_TO_ID_ROLE_ALL = {PAD_TOKEN: 0, UNK_TOKEN: 1, NEGATIVE_LABEL_ROLE: 2, 'Buyer': 3, 'Target': 4,
                    'Instrument': 5, 'Person': 6, 'Victim': 7,
                    'Time-Holds': 8, 'Attacker': 9, 'Time-Starting': 10, 'Price': 11, 'Time-Before': 12,
                    'Artifact': 13, 'Time-After': 14, 'Sentence': 15, 'Crime': 16, 'Destination': 17,
                    'Org': 18, 'Time-Ending': 19, 'Beneficiary': 20, 'Seller': 21, 'Time-At-End': 22,
                    'Recipient': 23, 'Position': 24, 'Money': 25, 'Giver': 26, 'Prosecutor': 27,
                    'Time-At-Beginning': 28, 'Entity': 29, 'Place': 30, 'Defendant': 31,
                    'Time-Within': 32, 'Plaintiff': 33, 'Adjudicator': 34, 'Agent': 35,
                    'Origin': 36, 'Vehicle': 37}

# NEGATIVE_LABEL = 'no_relation'
# LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3, 'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6, 'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9, 'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12, 'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15, 'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19, 'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23, 'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27, 'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30, 'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33, 'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36, 'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39, 'org:dissolved': 40, 'per:country_of_death': 41}

INFINITY_NUMBER = 1e12

# TYPE_ROLE_MAP : in encoder_ont
# not mask, zero-shot, so only list the ones that belongs to the ontology
