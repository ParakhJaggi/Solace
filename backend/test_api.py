"""
Simple test script for the backend API
Run the server first: python main.py
Then run this: python test_api.py
"""

import requests
import json

BASE_URL = "http://localhost:8080"

def test_health():
    """Test health check endpoint"""
    print("Testing /healthz...")
    response = requests.get(f"{BASE_URL}/healthz")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    return response.status_code == 200

def test_recommend(issue: str):
    """Test recommendation endpoint"""
    print(f"Testing /recommend with issue: '{issue}'")
    response = requests.post(
        f"{BASE_URL}/recommend",
        json={"issue": issue, "n": 3}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nFound {len(data['verses'])} verses:")
        for i, verse in enumerate(data['verses'], 1):
            print(f"\n{i}. {verse['ref']} (score: {verse['score']:.3f})")
            print(f"   {verse['text'][:150]}...")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "="*70 + "\n")
    return response.status_code == 200

def main():
    print("="*70)
    print("Bible Verse Companion API - Test Suite")
    print("="*70 + "\n")
    
    # Test health
    if not test_health():
        print("❌ Health check failed! Is the server running?")
        return
    
    print("✅ Health check passed!\n")
    
    # Test various queries
    test_cases = [
        "I'm anxious about work",
        "I feel guilty about a mistake I made",
        "I'm feeling lonely and isolated",
        "I need courage to face a difficult situation"
    ]
    
    for issue in test_cases:
        test_recommend(issue)
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    main()

