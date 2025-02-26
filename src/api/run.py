import uvicorn
import asyncio
import logging
import sys
import os
from pathlib import Path
import yaml
from multiprocessing import Process
from typing import Dict, Any

# Add src to path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from models.federated_client import FederatedClient
from models.fusion_model import FusionModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_api_server(host: str, port: int):
    """Run the FastAPI server"""
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

def run_federated_client(config: Dict[str, Any]):
    """Run the federated learning client"""
    model = FusionModel(config['model'])
    client = FederatedClient(model)
    
    async def federated_training_loop():
        while True:
            try:
                # Train for specified number of local epochs
                logger.info("Starting local training round")
                # TODO: Implement actual training loop with data loading
                await asyncio.sleep(60)  # Temporary placeholder
            except Exception as e:
                logger.error(f"Error in federated training: {e}")
                await asyncio.sleep(10)

    asyncio.run(federated_training_loop())

def main():
    """Main entry point"""
    try:
        config_path = os.path.join(src_path.parent, 'configs', 'default_config.yaml')
        config = load_config(config_path)
        
        # Start API server in a separate process
        api_process = Process(
            target=run_api_server,
            args=(config['api']['host'], config['api']['port'])
        )
        api_process.start()
        
        # Start federated client in a separate process
        fed_process = Process(
            target=run_federated_client,
            args=(config,)
        )
        fed_process.start()
        
        # Wait for processes to complete
        api_process.join()
        fed_process.join()
        
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        api_process.terminate()
        fed_process.terminate()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()