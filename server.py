from sanic import Sanic, response
from redis import Redis
from rq import Queue
from rq.job import Job


app = Sanic("CPR_Queue_Server")
redis_conn = Redis(host='redis', port=6379)
q = Queue(connection=redis_conn)

@app.post("/extract")
async def trigger_extraction(request):
    front = request.files.get("front")
    if not front:
        return response.json({"error": "Front image not provided"}, status=400)

    back = request.files.get("back")
    if not back:
        return response.json({"error": "Back image not provided"}, status=400)

    # Enqueue the task
    job = q.enqueue("task.process_cpr_task", front.body, back.body)
    
    return response.json({
        "job_id": job.id,
        "status_url": f"/status/{job.id}"
    }, status=202)

@app.get("/status/<job_id>")
async def get_status(_, job_id):
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        return response.json({"error": "Job not found"}, status=404)

    return response.json({
        "job_id": job_id,
        "status": job.get_status(),
        "result": job.result if job.is_finished else None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=1, access_log=False, dev=True)
