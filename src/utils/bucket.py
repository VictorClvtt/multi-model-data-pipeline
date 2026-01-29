import boto3
from botocore.exceptions import ClientError
import pandas as pd
from io import StringIO

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="admin",
        aws_secret_access_key="admin123",
        region_name="us-east-1"
    )

def save_to_minio(df: pd.DataFrame, bucket: str, key: str):
    s3 = get_minio_client()

    ensure_bucket_exists(s3, bucket)

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue()
    )

def ensure_bucket_exists(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        if error_code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=bucket)
        else:
            raise