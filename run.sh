#!/bin/bash

# Function to clean up background processes
cleanup() {
    echo "Cleaning up..."
    # Kill backend and frontend if they're running
    if [ ! -z "$BACKEND_PID" ]; then
        echo "Stopping backend server (PID: $BACKEND_PID)"
        kill $BACKEND_PID 2>/dev/null
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "Stopping frontend server (PID: $FRONTEND_PID)"
        kill $FRONTEND_PID 2>/dev/null
    fi
    
    exit 0
}

# Set up trap to call cleanup function when script is interrupted
trap cleanup INT TERM

# Check if required directories exist
if [ ! -d "app" ]; then
    echo "Error: 'app' directory not found. Please run this script from the project root."
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo "Error: 'frontend' directory not found. Please run this script from the project root."
    exit 1
fi

# Start backend server
echo "Starting backend server..."
cd app
python -m app.main &
BACKEND_PID=$!
cd ..

# Wait a moment to ensure backend starts properly
sleep 2

# Check if backend is still running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend server failed to start"
    cleanup
    exit 1
fi

echo "Backend server running with PID: $BACKEND_PID"
echo "API available at: http://localhost:8000/api/v1"

# Start frontend server
echo "Starting frontend server..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Wait a moment to ensure frontend starts properly
sleep 2

# Check if frontend is still running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "Error: Frontend server failed to start"
    cleanup
    exit 1
fi

echo "Frontend server running with PID: $FRONTEND_PID"
echo "Frontend available at: http://localhost:5173"

echo "MCP Agent is now running. Press Ctrl+C to stop."

# Wait for user to press Ctrl+C
wait