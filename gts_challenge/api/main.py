from fastapi import FastAPI
from .endpoints import router  # Import the router
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(router)  # Include the router in the app

@app.on_event("startup")
async def startup_event():
    logger.info("Loading models and pipelines at startup...")
