# import boto3
#
aws_access_key_id = 'AKIA3Y4NTXEG447DB24X'
aws_secret_access_key = '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'
bucket_region = 'US East (Ohio) us-east-2'
#
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                         region_name=bucket_region)

import pandas as pd

food_file = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')
food_names = food_file['ID']

paths = []

for food_name in food_names:
    first_letter = food_name[0]
    last_letter = food_name[-1]

    if first_letter == 'b':
        bucket_path = f"coco-nutrition-images/images100x100/{first_letter}/{last_letter}/{food_name}"
    else:
        bucket_path = f"coco-nutrition-images/images100x100/{first_letter}/{food_name}"

    paths.append(bucket_path)


output = pd.DataFrame({'Paths':paths, 'ID':food_names})

output.to_csv('path.csv')



# Extract the first and last letters from the food name
# first_letter = food_name[0]
# last_letter = food_name[-1]
#
# # Form the bucket path
# bucket_path = f"{first_letter}/{last_letter}/{food_name}"
#
# # Access the S3 bucket
# bucket_objects = s3_client.list_objects_v2(Bucket=bucket_path)
#
# # Process the objects in the bucket
# for obj in bucket_objects['Contents']:
#     print(obj['Key'])
