from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as transaction_router

# init FastAPI application instance
app = FastAPI()

# include router instance (i.e. endpoints)
app.include_router(transaction_router, prefix="/api/v1")

# Allow all origins, or restrict to specific origins if necessary
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://192.168.33.10:3000","http://192.168.33.10"],  # Allow React app origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
async def root():
    return {"message": "Welcome to the MongoDB Transaction API"}

