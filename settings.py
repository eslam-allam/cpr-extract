# settings.py
import os

# Set your Redis connection
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QUEUES = ["default"]

import task
