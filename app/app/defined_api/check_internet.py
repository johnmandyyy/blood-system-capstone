import subprocess

class Pinger(object):

    def __init__(self):
        pass

    @staticmethod
    def check_internet():
        """ A pinger for internet. """
        try:
            # Run ping command (works for Windows, macOS, Linux)
            result = subprocess.run(
                ["ping", "-t", "google.com"],  # use "-n" instead of "-c" for Windows
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3  # seconds
            )
            if result.returncode == 0:
                return True
            else:
                return False
        except subprocess.TimeoutExpired:
            return False
        except Exception as e:
            return False
        
