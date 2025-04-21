import json
from PIL import Image
import base64
import matplotlib.pyplot as plt


################################################################
# Helper funcitons for data preparation
################################################################

def get_image_from_s3_as_base64(bucket, image_path, image_size='small'):
    s3_object = bucket.Object(f"images/{image_size}/{image_path}")
    response = s3_object.get()
    file_stream = response['Body']
    return base64.b64encode(file_stream.read()).decode("utf-8")

def generate_embedding(bucket, text, image_path, bedrock_client, model_id):
    image_base64 = get_image_from_s3_as_base64(bucket, image_path)
    body = json.dumps(
        {
            "inputText": text,
            "inputImage": image_base64,
            "embeddingConfig": {
                "outputEmbeddingLength": 1024
            }
        }
    )
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body, 
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body["embedding"]

def filter_en_us(element):
    try:
        value = [item['value'] for item in element if item["language_tag"] == 'en_US']
        value = value[0] if value else None
    except:
        value = element
    return value

def filter_single_value(element):
    try:
        value = f"{element[0]['value']}"
    except:
        value = element
    return value

def filter_weight(element):
    try:
        value = f"{element[0]['normalized_value']['value']} {element[0]['normalized_value']['unit']}"
    except:
        value = element
    return value

def filter_dimensions(element):
    try:
        value = f"{element['length']['normalized_value']['value']} {element['length']['normalized_value']['unit']}, {element['width']['normalized_value']['value']} {element['width']['normalized_value']['unit']}, {element['height']['normalized_value']['value']} {element['height']['normalized_value']['unit']}"
    except:
        value = element
    return value

def filter_node_list(element):
    try:
        value = [item['node_name'] for item in element if item['node_name'] != None]
        value = ','.join(value)
    except:
        value = element
    return value

################################################################
# Helper funcitons for Query Workflow
################################################################

def get_image_from_s3(bucket, image_path, image_size='small'):
    s3_object = bucket.Object(f"images/{image_size}/{image_path}")
    response = s3_object.get()
    file_stream = response['Body']
    return Image.open(file_stream)

def show_results_list(results, limit=3):
    results = results[:limit]
    for result in results:
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print(f"Score: {round(result.get('_score'),4)} \t Item ID: {result.get('_id')}")
        print(f"Item Name: {result.get('_source').get('item_name')}")
        print(f"Fabric Type: {result.get('_source').get('fabric_type')}\t Material: {result.get('_source').get('material')} \t Color: {result.get('_source').get('color')}\t Style: {result.get('_source').get('style')}")
    print('--------------------------------------------------------------------------------------------------------------------------------')

def show_image_results(bucket, results, limit=3):
    results = results[:limit]
    fig, ax = plt.subplots(1, len(results), figsize=(3*limit, 2))
    for i, result in enumerate(results):  
        image = get_image_from_s3(bucket, result.get('_source').get('path'))
        ax = ax.ravel()        
        ax[i].imshow(image)
        ax[i].set_title(f"ID: {result.get('_id')}")
        ax[i].axis('off')
    plt.show()

