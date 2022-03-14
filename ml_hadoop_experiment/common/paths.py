from urllib import parse


def check_full_hdfs_path(path: str) -> bool:
    schema, netloc, path, _, _, _, = parse.urlparse(path)
    return len(netloc) > 0 and schema in ["hdfs", "viewfs"]
