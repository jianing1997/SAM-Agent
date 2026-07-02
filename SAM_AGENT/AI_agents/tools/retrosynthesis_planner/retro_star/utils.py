import logging

def setup_logger():
    """
    Setup logger for retro_star module
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

