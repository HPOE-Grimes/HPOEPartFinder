#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Navigate to the backend directory
cd backend

# Start the FastAPI server with uvicorn
uvicorn app:app --reload