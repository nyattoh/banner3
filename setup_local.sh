#!/bin/bash

# Banner Layer Decomposition - Local Development Setup
# Worker3 Implementation

echo "🚀 Setting up Banner Layer Decomposition - Worker3"
echo "=================================================="

# Check if running on Ubuntu/Debian
if command -v apt-get &> /dev/null; then
    echo "📦 Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install Tesseract OCR
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-jpn
    
    # Install OpenCV dependencies
    sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
    
    # Install Python development tools
    sudo apt-get install -y python3-dev python3-pip python3-venv
    
    echo "✅ System dependencies installed"
else
    echo "⚠️  Non-Debian system detected. Please manually install:"
    echo "   - Tesseract OCR (with English and Japanese language packs)"
    echo "   - OpenCV system dependencies"
    echo "   - Python development tools"
fi

# Create Python virtual environment
echo "🐍 Setting up Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Backend dependencies installed"

# Setup frontend
echo "⚛️  Setting up React frontend..."
cd ../frontend
npm install

echo "✅ Frontend dependencies installed"

# Create necessary directories
echo "📁 Creating necessary directories..."
cd ../backend
mkdir -p uploads static temp logs

echo "🎉 Local setup complete!"
echo ""
echo "To start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo "2. Frontend: cd frontend && npm start"
echo ""
echo "Access URLs:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"