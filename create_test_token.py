#!/usr/bin/env python3
"""
Create a test JWT token for API authentication.
"""

import sys
import os
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from jose import jwt
    from src.utils.config import settings
    
    # Create test token
    data = {
        "sub": "test_user",
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    
    token = jwt.encode(data, settings.secret_key, algorithm=settings.algorithm)
    
    print("=" * 60)
    print("Test JWT Token Generated")
    print("=" * 60)
    print(f"\nToken:\n{token}\n")
    print("=" * 60)
    print("\nUsage:")
    print(f'curl -H "Authorization: Bearer {token}" \\')
    print('     http://localhost:8000/api/v1/predictions/SPY')
    print("\n" + "=" * 60)
    
except ImportError as e:
    print("Error: Missing dependencies. Install with: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

