from fastapi import FastAPI

app = FastAPI(title="ISL Communicator API")

@app.get("/")
def root():
    return {"status": "backend is running"}