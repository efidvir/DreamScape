from cryptography.fernet import Fernet

# Generate secret encryption key (run this ONLY ONCE)
key = Fernet.generate_key()
with open('dec.key', 'wb') as key_file:
    key_file.write(key)

# Encrypt your original JSON key file
with open('dreamscape-454613-b21d1604174d.json', 'rb') as original_file:
    original_json = original_file.read()

encrypted_json = Fernet(key).encrypt(original_json)

# Save encrypted JSON file
with open('pyencrypted_key.json', 'wb') as encrypted_file:
    encrypted_file.write(encrypted_json)

print("🔐 Encryption successful. Files 'dec.key' and 'pyencrypted_key.json' created.")
