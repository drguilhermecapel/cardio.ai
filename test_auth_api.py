import requests
import json


def test_authentication():
    """Test authentication API directly"""
    try:
        health_response = requests.get("http://localhost:8000/health")
        print(f"Health check: {health_response.status_code} - {health_response.text}")

        login_data = {"username": "admin", "password": "admin"}

        login_response = requests.post(
            "http://localhost:8000/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        print(f"Login attempt: {login_response.status_code}")
        print(f"Response: {login_response.text}")

        if login_response.status_code == 200:
            print("✅ Authentication successful")
            return True
        else:
            print("❌ Authentication failed")
            return False

    except Exception as e:
        print(f"Error testing authentication: {e}")
        return False


if __name__ == "__main__":
    success = test_authentication()
    print(f"Authentication test result: {'PASS' if success else 'FAIL'}")
