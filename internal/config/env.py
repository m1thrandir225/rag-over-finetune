import os


def get_required_env_var(env_var: str) -> str:
    """
    Returns the value of a required environment variable
    """
    value = os.getenv(env_var)
    if not value:
        raise ValueError(
            f"Missing API key: {env_var} environment variable is not set. "
            f"Please add it to your .env file."
        )
    return value
