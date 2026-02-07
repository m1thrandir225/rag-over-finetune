import torch


def _detect_best_device() -> str:
    """
    Auto detect the best available device for embeddings

    CUDA > MPS > CPU
    """

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
