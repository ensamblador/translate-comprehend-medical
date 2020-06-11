import boto3
import sys
import json
translate = boto3.client('translate')
cmedical = boto3.client('comprehendmedical')

if len(sys.argv) > 1:
    with open(sys.argv[1]) as f:
        text = f.read()
else:
    print ('indique archivo con texto')
    exit

traducido = translate.translate_text( Text=text, SourceLanguageCode='es', TargetLanguageCode='en')['TranslatedText']


entities_response = cmedical.detect_entities_v2(Text=traducido)
entities = entities_response['Entities']
unnmaped_attr = entities_response['UnmappedAttributes']


phi_response = cmedical.detect_phi( Text=traducido)
phi_entities = phi_response['Entities']

icd10_response = cmedical.infer_icd10_cm(Text=traducido)
icd10_entities = icd10_response['Entities']

result = dict(
 texto_original=text,
 texto_traducido=traducido,
 entites = entities,
 unnmaped_attributes = unnmaped_attr,
 phi_entities = phi_entities,
 icd10_entities = icd10_entities
)


print (json.dumps(result))