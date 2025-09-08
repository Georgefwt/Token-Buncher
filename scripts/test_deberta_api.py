#!/usr/bin/env python3
import requests
import json
import sys
import time

def test_api(url="http://localhost:5000"):
    """Test the DeBERTa classifier API with example texts"""
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{url}/health")
        print(f"Health check: {health_response.json()}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {url}")
        print("Make sure the API server is running.")
        sys.exit(1)
    
    # Sample texts to test
    sample_texts = [
        "I'm sorry, I cannot assist with that."
    ]
    
    # Prepare request payload
    payload = {"texts": sample_texts}
    
    # Time the request
    start_time = time.time()
    
    # Send request to score endpoint
    try:
        response = requests.post(
            f"{url}/score", 
            json=payload,
            headers={"Content-Type": "application/json"}
        )
    except requests.exceptions.ConnectionError:
        print(f"Error: Failed to connect to the scoring endpoint at {url}/score")
        sys.exit(1)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        print("result: ", result)
        
        print(f"\nScored {result['count']} texts in {elapsed_time:.2f} seconds")
        print("\nResults:")
        
        # Print each text with its score
        for i, (text, score) in enumerate(zip(sample_texts, result['scores'])):
            # Truncate text if too long for display
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"Text {i+1}: \"{display_text}\"")
            print(f"Score: {score:.4f}")
            print("-" * 50)
            
        # Calculate average score
        avg_score = sum(result['scores']) / len(result['scores'])
        print(f"\nAverage score: {avg_score:.4f}")
        
    else:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    # Get custom URL from command line argument if provided
    api_url = "http://localhost:50050"
    test_api(api_url) 