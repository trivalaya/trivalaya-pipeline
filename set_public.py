import boto3
import json
# Use the same credentials your pipeline uses
session = boto3.session.Session()
client = session.client('s3',
                        region_name='sfo3',
                        endpoint_url='https://sfo3.digitaloceanspaces.com',
                        aws_access_key_id='DO004YCRXKWM2CXJB78J',
                        aws_secret_access_key='j5D/Az7RIXUn1W3y2rxmZbCmgj/PH5G9vj+tOIJWCRU')

bucket_name = 'trivalaya-data'

policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "PublicReadCrops",
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": f"arn:aws:s3:::{bucket_name}/processed/vision/v1/crops/*"
    }]
}

client.put_bucket_policy(Bucket=bucket_name, Policy=json.dumps(policy))
print("Bucket policy applied. The crops folder is now public.")