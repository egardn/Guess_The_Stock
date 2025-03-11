import os
import sys
import argparse

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
import uvicorn
from pyngrok import ngrok
import nest_asyncio

def run_api(port=8000, use_ngrok=False, ngrok_token=None):
    """Run the FastAPI application"""
    if use_ngrok:
        if not ngrok_token:
            print("Warning: ngrok token not provided. Using default settings.")
        else:
            ngrok.set_auth_token(ngrok_token)
            
        # Apply the nest_asyncio patch
        nest_asyncio.apply()

        # Create a tunnel to the localhost
        public_url = ngrok.connect(port)
        print(f"Public URL: {public_url}")
    
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Order Book Prediction API")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--ngrok", action="store_true", help="Use ngrok to expose the API")
    parser.add_argument("--ngrok-token", type=str, help="Ngrok auth token")
    
    args = parser.parse_args()
    
    run_api(port=args.port, use_ngrok=args.ngrok, ngrok_token=args.ngrok_token)