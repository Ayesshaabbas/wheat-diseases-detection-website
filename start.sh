#!/bin/bash

echo "🚀 Starting WheatGuard"

# Start backend
echo "📦 Starting backend..."
cd backend
python create_model.py
python app.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting frontend..."
cd ..
npm run dev &
FRONTEND_PID=$!

echo "✅ Both servers started!"
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend: http://localhost:5000"

# Wait for user to stop
read -p "Press Enter to stop servers..."

# Kill both processes
kill $BACKEND_PID $FRONTEND_PID
echo "🛑 Servers stopped"
