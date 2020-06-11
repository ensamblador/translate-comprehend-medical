#%%
import boto3
import pandas as pd
translate = boto3.client('translate')
cmedical = boto3.client('comprehendmedical')
# %%

text = ""

with open("diagnostico.txt") as f:
        text = f.read()

# %%
response = translate.translate_text(
    Text=text,
    SourceLanguageCode='es',
    TargetLanguageCode='en'
)
traducido = response['TranslatedText']

# %%
response = cmedical.detect_entities_v2(
    Text=traducido
)
entities = response['Entities']
unnmaped_attr = response['UnmappedAttributes']
# %%

# DETECCION DE ENTIDADES
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehendmedical.html#ComprehendMedical.Client.detect_entities_v2

entites_df = pd.DataFrame(entities)
# %%
for ent in entities:
    if len(ent["Traits"]):
        print (ent["Category"],":",ent["Text"],",", ent["Traits"][0]['Name'], " score:",ent["Traits"][0]['Score'])
# %%
entites_df[entites_df.Score>0.7]

# %%
for attr in unnmaped_attr:
    print (attr['Type'],':\n****************************\n',attr['Attribute'],'\n')

# %%
# DETECCCION PHI (protected health information)
response = cmedical.detect_phi(
    Text=traducido
)
response['Entities']
# %%

#ICD-10-CM
#https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehendmedical.html#ComprehendMedical.Client.infer_icd10_cm


response = cmedical.infer_icd10_cm(
    Text=traducido
)

ICD10CM = response['Entities']

# %%
ICD_df =pd.DataFrame(ICD10CM)
# %%
ICD_df

# %%
# InferRxNorm
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/comprehendmedical.html#ComprehendMedical.Client.infer_rx_norm

response = cmedical.infer_rx_norm(
    Text=traducido
)

rxnorm = response['Entities']

# %%
# Pricing
# https://aws.amazon.com/comprehend/pricing/
'''
Feature	Price per unit
Medical Named Entity and Relationship Extraction (NERe) API	$0.01
Medical Protected Health Information Data Extraction and Identification (PHId) API	$0.0014
Medical ICD-10-CM Ontology Linking API	$0.0005
Medical RxNORM Ontology Linking API	$0.00025
Amazon Comprehend Medical Free Tier

Amazon Comprehend Medical offers a free tier covering 25k units of text (2.5M characters) for the first three months when you start using the service for any of the APIs.
'''

# ejemplo: 