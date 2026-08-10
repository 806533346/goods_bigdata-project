"""
云端存储 I/O 工具 — 兼容阿里云 OSS 和 AWS S3。

使用方式:
  # 上传
  upload_to_cloud("data/train.csv", "raw/train.csv")

  # 下载
  download_from_cloud("raw/train.csv", "data/train.csv")

  # 上传模型检查点
  upload_checkpoint("model.pt", version="best")

  # 同步整个数据目录
  sync_data_to_local()

底层自动选择:
  1. boto3 (S3 API, 阿里云 OSS 兼容)
  2. oss2   (阿里云原生 SDK, fallback)

使用前设置环境变量:
  export OSS_ENDPOINT=oss-cn-hangzhou.aliyuncs.com
  export OSS_BUCKET=comp5434-bucket
  export OSS_ACCESS_KEY_ID=xxx
  export OSS_ACCESS_KEY_SECRET=xxx
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports — only load the SDK actually needed
_s3_client = None
_oss_bucket_obj = None


def get_s3_client():
    """Return a boto3 S3 client configured for the active cloud provider."""
    global _s3_client
    if _s3_client is not None:
        return _s3_client

    from config import OSS_ENDPOINT, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, CLOUD_PROVIDER

    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        raise ImportError("boto3 is required for cloud I/O. Install with: pip install boto3")

    if CLOUD_PROVIDER == "aliyun":
        endpoint_url = f"https://{OSS_ENDPOINT}"
    else:
        endpoint_url = None  # Use default AWS endpoint

    _s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=OSS_ACCESS_KEY_ID or None,
        aws_secret_access_key=OSS_ACCESS_KEY_SECRET or None,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    logger.info(f"S3 client initialized for provider={CLOUD_PROVIDER}, endpoint={endpoint_url}")
    return _s3_client


def download_from_cloud(cloud_key: str, local_path: str, bucket: Optional[str] = None) -> str:
    """
    Download a file from cloud object storage to local disk.

    Args:
        cloud_key: Object key in bucket (e.g. 'raw/train.csv')
        local_path: Local file path to save to
        bucket: Override default bucket

    Returns:
        Absolute path to the downloaded file
    """
    from config import OSS_BUCKET

    bucket = bucket or OSS_BUCKET
    local_path = str(local_path)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    logger.info(f"Downloading: oss://{bucket}/{cloud_key} → {local_path}")

    try:
        s3 = get_s3_client()
        s3.download_file(bucket, cloud_key, local_path)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        # Fallback: try Alibaba Cloud native SDK
        _download_via_oss2(cloud_key, local_path, bucket)

    logger.info(f"Download complete: {local_path} ({os.path.getsize(local_path):,} bytes)")
    return os.path.abspath(local_path)


def upload_to_cloud(local_path: str, cloud_key: str, bucket: Optional[str] = None) -> str:
    """
    Upload a local file to cloud object storage.

    Args:
        local_path: Path to local file
        cloud_key: Destination object key in bucket
        bucket: Override default bucket

    Returns:
        Cloud URI string (e.g. oss://bucket/key)
    """
    from config import OSS_BUCKET

    bucket = bucket or OSS_BUCKET
    local_path = str(local_path)

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    logger.info(f"Uploading: {local_path} → oss://{bucket}/{cloud_key}")

    try:
        s3 = get_s3_client()
        s3.upload_file(local_path, bucket, cloud_key)
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        _upload_via_oss2(local_path, cloud_key, bucket)

    logger.info(f"Upload complete: oss://{bucket}/{cloud_key}")
    return f"oss://{bucket}/{cloud_key}"


def list_cloud_objects(prefix: str = "", bucket: Optional[str] = None) -> list:
    """List objects in cloud bucket with given prefix."""
    from config import OSS_BUCKET

    bucket = bucket or OSS_BUCKET
    s3 = get_s3_client()

    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"key": obj["Key"], "size": obj["Size"]})
    return objects


def cloud_file_exists(cloud_key: str, bucket: Optional[str] = None) -> bool:
    """Check if a file exists in cloud storage."""
    from config import OSS_BUCKET

    bucket = bucket or OSS_BUCKET
    try:
        s3 = get_s3_client()
        s3.head_object(Bucket=bucket, Key=cloud_key)
        return True
    except Exception:
        return False


def sync_data_to_local(data_dir: Optional[str] = None):
    """
    Download all required data files from cloud to local cache.
    Skips files that already exist locally.
    """
    from config import (
        DATA_DIR, OSS_TRAIN_CSV, OSS_TEST_CSV, OSS_PRODINFO_CSV,
        OSS_USER_STATS, OSS_PROD_STATS, OSS_PARENT_STATS, OSS_GLOBAL_AVG,
    )

    data_dir = Path(data_dir or DATA_DIR)
    files_to_sync = [
        (OSS_TRAIN_CSV, data_dir / "train.csv"),
        (OSS_TEST_CSV, data_dir / "test.csv"),
        (OSS_PRODINFO_CSV, data_dir / "prodInfo.csv"),
        (OSS_USER_STATS, data_dir / "user_stats.parquet"),
        (OSS_PROD_STATS, data_dir / "prod_stats.parquet"),
        (OSS_PARENT_STATS, data_dir / "parent_stats.parquet"),
        (OSS_GLOBAL_AVG, data_dir / "global_avg.npy"),
    ]

    for cloud_key, local_path in files_to_sync:
        if local_path.exists():
            logger.info(f"Skipping (exists): {local_path}")
            continue
        try:
            download_from_cloud(cloud_key, str(local_path))
        except Exception as e:
            logger.warning(f"Could not download {cloud_key}: {e}")


def upload_checkpoint(local_path: str, version: str = "latest"):
    """Upload a model checkpoint to cloud."""
    from config import OSS_CHECKPOINTS
    filename = os.path.basename(local_path)
    cloud_key = f"{OSS_CHECKPOINTS}/{version}/{filename}"
    return upload_to_cloud(local_path, cloud_key)


def download_checkpoint(version: str = "best", local_dir: Optional[str] = None):
    """Download a model checkpoint from cloud."""
    from config import OSS_CHECKPOINTS, DATA_DIR
    local_dir = local_dir or str(DATA_DIR)
    cloud_key = f"{OSS_CHECKPOINTS}/{version}/roberta_base_finetuned.pt"
    local_path = os.path.join(local_dir, f"roberta_base_finetuned_{version}.pt")
    return download_from_cloud(cloud_key, local_path)


# ── Alibaba Cloud native SDK fallbacks ──────────────────────────────────

def _get_oss_bucket():
    """Lazy-init Alibaba Cloud OSS native bucket object."""
    global _oss_bucket_obj
    if _oss_bucket_obj is not None:
        return _oss_bucket_obj

    from config import OSS_ENDPOINT, OSS_BUCKET, OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET

    try:
        import oss2
    except ImportError:
        raise ImportError("oss2 is required. Install with: pip install oss2")

    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    _oss_bucket_obj = oss2.Bucket(auth, f"https://{OSS_ENDPOINT}", OSS_BUCKET)
    return _oss_bucket_obj


def _download_via_oss2(cloud_key: str, local_path: str, bucket: str):
    """Fallback: download using Alibaba Cloud OSS native SDK."""
    try:
        b = _get_oss_bucket()
        b.get_object_to_file(cloud_key, local_path)
        logger.info(f"Downloaded via oss2: {cloud_key}")
    except Exception as e:
        raise RuntimeError(f"Both S3 and oss2 download failed for {cloud_key}: {e}")


def _upload_via_oss2(local_path: str, cloud_key: str, bucket: str):
    """Fallback: upload using Alibaba Cloud OSS native SDK."""
    try:
        b = _get_oss_bucket()
        b.put_object_from_file(cloud_key, local_path)
        logger.info(f"Uploaded via oss2: {cloud_key}")
    except Exception as e:
        raise RuntimeError(f"Both S3 and oss2 upload failed for {cloud_key}: {e}")
