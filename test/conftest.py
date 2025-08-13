import psutil


def is_mem_limited():
    """Check if the current environment has limited memory.

    Returns:
        bool: True if memory is limited (less than 16 GiB), False otherwise.
    """
    # Check if we are running on a limited memory host (e.g. github action)
    total_mem_gib = psutil.virtual_memory().total // 1024**3
    return total_mem_gib < 16
