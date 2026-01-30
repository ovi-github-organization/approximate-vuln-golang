from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

def generate_key():
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)

    ciphertext = cipher_suite.encrypt(data)
    decrypted_text = cipher_suite.decrypt(ciphertext)

def signature_key():
    signature = private_key.sign(
    data,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
    )

    public_key.verify(
    signature,
    data,
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256() 
    )   

if __name__ == '__main__':
    data = b"Hello, World!"
    private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048)
    public_key = private_key.public_key()
    generate_key()
    signature_key()