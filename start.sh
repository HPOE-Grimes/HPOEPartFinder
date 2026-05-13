#!/bin/bash

# Start frontend static server in the background
python3 -m http.server 5500 --directory frontend &
FRONTEND_PID=$!

# Activate the virtual environment
source venv/bin/activate

# Navigate to the backend directory
cd backend

# Start the FastAPI server with uvicorn
uvicorn app:app --reload

# Kill frontend server when backend exits
kill $FRONTEND_PID