"""
Sample Vulnerable Code for Testing

This file contains intentional security vulnerabilities
for demonstrating the code review system.

DO NOT USE IN PRODUCTION!
"""

import os
import pickle
import yaml
import hashlib
import random
import subprocess

# CWE-798: Hardcoded Credentials
SECRET_KEY = "super_secret_key_123"
DATABASE_PASSWORD = "admin123"
API_KEY = "sk-1234567890abcdef1234567890abcdef"


class AuthService:
    """Authentication service with multiple vulnerabilities."""
    
    def __init__(self):
        self.db_password = "hardcoded_password"  # CWE-798
    
    def login(self, username, password):
        """Vulnerable login function."""
        # CWE-89: SQL Injection
        query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
        
        # Execute query (simulated)
        print(f"Executing: {query}")
        
        # CWE-328: Weak Hash
        password_hash = hashlib.md5(password.encode()).hexdigest()
        
        return {"user": username, "hash": password_hash}
    
    def execute_command(self, user_input):
        """Vulnerable command execution."""
        # CWE-78: Command Injection
        os.system(f"echo {user_input}")
        
        # Also vulnerable
        subprocess.run(f"ls {user_input}", shell=True)
    
    def process_data(self, data):
        """Vulnerable deserialization."""
        # CWE-502: Insecure Deserialization
        return pickle.loads(data)
    
    def load_config(self, config_file):
        """Vulnerable YAML loading."""
        with open(config_file) as f:
            # CWE-502: Unsafe YAML load
            return yaml.load(f)
    
    def generate_token(self):
        """Insecure random generation."""
        # CWE-330: Insecure Random
        token = random.randint(1000, 9999)
        return str(token)
    
    def log_user_action(self, user, action, password):
        """Logging sensitive data."""
        # CWE-532: Sensitive data in logs
        print(f"User {user} performed {action} with password {password}")


def evaluate_expression(user_input):
    """Vulnerable eval usage."""
    # CWE-95: Code Injection via eval
    result = eval(user_input)
    return result


def render_html(user_content):
    """Vulnerable to XSS (conceptual)."""
    # In a real app, this would be dangerous
    html = f"<div>{user_content}</div>"
    return html


# Bare except
def risky_operation():
    try:
        dangerous_thing()
    except:  # Bad: catches everything
        pass


# Unused variable
unused_variable = "I'm never used"

def unused_function():
    """This function is never called."""
    pass


# Function too long (simplified example)
def very_long_function(a, b, c, d, e, f, g):
    """Function with too many parameters."""
    result = a + b
    result = result + c
    result = result + d
    result = result + e
    result = result + f
    result = result + g
    # ... imagine 50 more lines ...
    return result


# Debug mode
DEBUG = True


if __name__ == "__main__":
    auth = AuthService()
    auth.login("admin", "password123")
