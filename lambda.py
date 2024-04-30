import json
import logging
import boto3

# Get the service resource.

client = boto3.client('dynamodb')
class_dict = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
def lambda_handler(event, context):
    # TODO implement
    threshold=0.9
    # print(event)
    for rec in event['Records']:
        # print(rec)
        if rec['eventName'] == 'INSERT':
            UpdateItem = rec['dynamodb']['NewImage']
            # print(UpdateItem)

            # lab4 code goes here
            features = UpdateItem['Features']['S']
            predicted_classs = UpdateItem['Class']['S']
            actual_class = UpdateItem['Actual']['S']
            probability = UpdateItem['Probability']['S']

            # Convert strings to their respective data types
            predicted_class = int(predicted_classs)
            actual_class =  class_dict.get(actual_class, -1)  # -1 for unknown class
            probability = float(probability)

            if predicted_class != actual_class or probability < threshold:
            # Copy the record to the 'IrisExtendedRetrain' table
                response = client.put_item(
                    TableName='IrisExtendedRetrain',
                    Item=UpdateItem
                )

            # response = client.put_item( TableName =  'IrisExtendedRetrain', Item = UpdateItem )
            # print (response)

    return {
        'statusCode': 200,
        'body': json.dumps( 'IrisExtendedRetrain Lambda return' )
    }
    
