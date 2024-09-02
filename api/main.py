from fastapi import FastAPI
from api.routes import router as transaction_router

app = FastAPI()

app.include_router(transaction_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the MongoDB Transaction API"}

# Run with: uvicorn backend.api.main:app --reload
