import hmac
import time
import base64
import hashlib
import struct
import random

class OTPHelpers:

    def __init__(self):
        """An OTP Helper for OTPs, it can be used for different use cases."""
        pass
    
    # --------------------------------------------------------------------------------------------------------------- #
    def generate_6_digit_code(self):
        """Generate a random 6 digit number, can be used in verifications."""
        
        code = random.randint(0, 999999)
        return f"{code:06}"

    # --------------------------------------------------------------------------------------------------------------- #
    def to_base32(self, input_string):
        """Convert string to base 64"""
        
        byte_string = input_string.encode("utf-8")
        base32_encoded = base64.b32encode(byte_string)
        base32_string = base32_encoded.decode("utf-8")
        return base32_string

    # --------------------------------------------------------------------------------------------------------------- #
    def generate_numeric_totp(self, secret, interval=5):
        """Generate a random 6 digit number that is purely code timebased, can be used in verifications."""
        
        key = base64.b32decode(secret.upper())
        current_time = int(time.time() // interval)
        msg = struct.pack(">Q", current_time)
        hmac_hash = hmac.new(key, msg, hashlib.sha1).digest()
        offset = hmac_hash[-1] & 0x0F
        truncated_hash = hmac_hash[offset : offset + 4]
        code = struct.unpack(">I", truncated_hash)[0] & 0x7FFFFFFF
        totp = code % 1000000
        return str(totp).zfill(6)
