from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title=os.getenv("APP_NAME", "ai-backend"))

# Simple CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers
from app.routes.health import router as health_router
from app.routes.analyze import router as analyze_router
from app.routes.fetch import router as fetch_router

app.include_router(health_router, prefix="/health")
app.include_router(analyze_router, prefix="/analyze")
app.include_router(fetch_router, prefix="/fetch")


@app.on_event("startup")
async def startup_event():
    print("Starting app — environment:", os.getenv("APP_NAME"))
