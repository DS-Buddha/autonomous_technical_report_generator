"""
FastAPI web application for the Hybrid Agentic System.
Provides a web interface for generating technical reports.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import queue

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from main import generate_report
from src.utils.logger import setup_logger
from src.utils.progress_streamer import get_progress_streamer, ProgressEvent

# Setup
app = FastAPI(
    title="Hybrid Agentic System",
    description="Autonomous Technical Report Generator",
    version="1.0.0"
)

# Setup logging
logger = setup_logger(__name__, log_file=Path('outputs/app.log'))

# Templates and static files
templates = Jinja2Templates(directory="templates")

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create outputs directory if it doesn't exist
outputs_dir = Path("outputs/reports")
outputs_dir.mkdir(parents=True, exist_ok=True)


# Request models
class ReportRequest(BaseModel):
    topic: str
    depth: str = "comprehensive"
    code_examples: bool = True
    max_iterations: int = 3
    report_mode: str = "staff_ml_engineer"


class ReportStatus(BaseModel):
    status: str
    message: str
    report_path: Optional[str] = None
    error: Optional[str] = None


# Store for tracking report generation status
report_jobs = {}

# Store for SSE connections
sse_queues = []

# Get progress streamer singleton
progress_streamer = get_progress_streamer(enable_console_output=True)

# Subscribe to progress events and broadcast to SSE clients
def broadcast_progress_event(event: ProgressEvent):
    """Broadcast progress event to all SSE clients."""
    event_data = event.to_json()

    # Send to all connected SSE clients
    dead_queues = []
    for q in sse_queues:
        try:
            q.put_nowait(event_data)
        except:
            dead_queues.append(q)

    # Clean up dead queues
    for q in dead_queues:
        if q in sse_queues:
            sse_queues.remove(q)

progress_streamer.subscribe(broadcast_progress_event)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the main page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/progress-stream")
async def progress_stream(request: Request):
    """
    Server-Sent Events endpoint for real-time progress updates.
    """
    async def event_generator():
        # Create a queue for this client
        q = queue.Queue()
        sse_queues.append(q)

        try:
            while True:
                # Check if client is still connected
                if await request.is_disconnected():
                    break

                # Try to get event from queue (non-blocking)
                try:
                    event_data = q.get_nowait()
                    yield f"data: {event_data}\n\n"
                except queue.Empty:
                    # Send keepalive comment every 15 seconds
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)

        finally:
            # Clean up when client disconnects
            if q in sse_queues:
                sse_queues.remove(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/api/generate", response_model=ReportStatus)
async def generate_report_api(report_request: ReportRequest):
    """
    Generate a technical report based on the provided parameters.
    """
    try:
        logger.info(f"Received report request for topic: {report_request.topic}")

        # Generate unique job ID
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{report_request.topic[:20].replace(' ', '_')}"

        # Update job status
        report_jobs[job_id] = {
            "status": "processing",
            "topic": report_request.topic,
            "started_at": datetime.now().isoformat()
        }

        # Run report generation in background
        async def run_generation():
            try:
                # Run the synchronous generate_report function in a thread pool
                loop = asyncio.get_event_loop()
                output_path = await loop.run_in_executor(
                    None,
                    generate_report,
                    report_request.topic,
                    report_request.depth,
                    report_request.code_examples,
                    report_request.max_iterations,
                    report_request.report_mode
                )

                report_jobs[job_id].update({
                    "status": "completed",
                    "report_path": str(output_path),
                    "completed_at": datetime.now().isoformat()
                })

            except Exception as e:
                logger.error(f"Report generation failed for job {job_id}: {e}", exc_info=True)
                report_jobs[job_id].update({
                    "status": "failed",
                    "error": str(e),
                    "completed_at": datetime.now().isoformat()
                })

        # Start generation in background
        asyncio.create_task(run_generation())

        return ReportStatus(
            status="started",
            message=f"Report generation started for topic: {report_request.topic}",
            report_path=None
        )

    except Exception as e:
        logger.error(f"Failed to start report generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a report generation job.
    """
    if job_id not in report_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return report_jobs[job_id]


@app.get("/api/reports")
async def list_reports():
    """
    List all generated reports.
    """
    try:
        reports_dir = Path("outputs/reports")
        reports = []

        if reports_dir.exists():
            for report_file in reports_dir.glob("*.md"):
                stat = report_file.stat()
                reports.append({
                    "filename": report_file.name,
                    "path": str(report_file),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

        # Sort by modified date (newest first)
        reports.sort(key=lambda x: x["modified_at"], reverse=True)

        return {"reports": reports}

    except Exception as e:
        logger.error(f"Failed to list reports: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report/{filename}")
async def get_report(filename: str):
    """
    Get the content of a specific report.
    """
    try:
        report_path = Path("outputs/reports") / filename

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        # Security check: ensure the file is within the reports directory
        if not str(report_path.resolve()).startswith(str(Path("outputs/reports").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            "filename": filename,
            "content": content
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read report {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{filename}")
async def download_report(filename: str):
    """
    Download a specific report file.
    """
    try:
        report_path = Path("outputs/reports") / filename

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        # Security check
        if not str(report_path.resolve()).startswith(str(Path("outputs/reports").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        return FileResponse(
            path=report_path,
            filename=filename,
            media_type='text/markdown'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download report {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/report/{filename}")
async def delete_report(filename: str):
    """
    Delete a specific report file.
    """
    try:
        report_path = Path("outputs/reports") / filename

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        # Security check
        if not str(report_path.resolve()).startswith(str(Path("outputs/reports").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")

        report_path.unlink()

        return {"message": f"Report {filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete report {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    import sys

    print("=" * 60)
    print("Hybrid Agentic System - Web Interface")
    print("=" * 60)
    print("\nStarting server on http://localhost:8001")
    print("\nPress CTRL+C to stop the server\n")

    # Use direct app object instead of string to avoid Windows compatibility issues
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
