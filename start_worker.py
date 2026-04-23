import redis
from rq import Worker
import worker_init

r = redis.Redis(host='redis')


worker_init.initialize()  # <-- load BEFORE worker.work() forks
worker = Worker(queues=["default"], connection=r)
worker.work()
