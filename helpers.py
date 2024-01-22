import os
import sys
import subprocess


def set_cwd():
    """
    Find '.repo-root' file.

    Searches parent folders starting with path of __file__ or current cwd if that fails.
    """
    # first, resolve the path of the file that calls this function or fall back to the
    # working directory if __file__ is not defined/an interactive session
    try:
        path = os.path.abspath(__file__)
    except NameError:
        # __file__ does not exist in a REPL
        path = os.getcwd()
    # guards against running in an IDE such as Jupyter or Pycharm
    # (where __file__ is not defined or "<some-random-string-input>")
    if "input>" in path:
        path = os.getcwd()

    # then, search for the .repo-root file. go at most 8 levels up
    for _ in range(8):
        # set the current working directory to the path of the .repo-root file
        if os.path.exists(os.path.join(path, ".repo-root")):
            os.chdir(path)
            return
        # ... or go one level up if the .repo-root file is not found
        else:
            path = os.path.dirname(path)


def ensure_requirements():
    """
    Ensure that the requirements are installed.

    Be sure to call set_cwd() first, or you might not find the requirements.txt file.
    """
    if not os.path.exists("requirements.txt"):
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )


def load_secrets(secrets_file: str = ".env"):
    """
    Load the secrets from the .env file.

    Must have one the following formats:
     - KEY=VALUE, one per line.
     - export KEY=VALUE, one per line.
    where VALUE is allowed to be surrounded by single or double quotes.
    """
    dotenv_file_contents = open(secrets_file).read()
    for line in dotenv_file_contents.splitlines():
        key, value = line.split("=")

        # remove 'export ' from the beginning of the key
        # this allows reading in source-able env files
        key = key.replace("export ", "", 1)

        # remove surrounding single or double quotes from the value
        value = value.strip("\"'")

        os.environ[key] = value


def init():
    """Initialize the environment."""
    set_cwd()
    # ensure_requirements()
    load_secrets()
