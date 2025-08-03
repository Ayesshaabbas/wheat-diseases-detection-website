# WheatGuard Setup Instructions

## Quick Start

### 1. Install Dependencies

**Frontend (Node.js):**
\`\`\`bash
npm install
\`\`\`

**Backend (Python):**
\`\`\`bash
cd backend
pip install -r requirements.txt
\`\`\`

### 2. Create Dummy Model (for testing)

\`\`\`bash
cd backend
python setup_dummy_model.py
\`\`\`

### 3. Start the Application

**Terminal 1 - Backend:**
\`\`\`bash
cd backend
python app.py
\`\`\`

**Terminal 2 - Frontend:**
\`\`\`bash
npm run dev
\`\`\`

### 4. Access the Application

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

## Troubleshooting

### Backend Issues

1. **"TensorFlow not available"**
   \`\`\`bash
   pip install tensorflow
   \`\`\`

2. **"Model file not found"**
   \`\`\`bash
   cd backend
   python setup_dummy_model.py
   \`\`\`

3. **Port 5000 already in use**
   - Change the port in `backend/app.py` (line with `app.run`)
   - Update `BACKEND_URL` in `app/utils/api.ts`

### Frontend Issues

1. **"Cannot connect to backend"**
   - Make sure Flask server is running
   - Check if port 5000 is accessible
   - Try: `curl http://localhost:5000/api/health`

2. **CORS errors**
   - Backend is configured to allow all origins
   - Check browser console for specific errors

### Testing the Setup

1. **Test backend directly:**
   \`\`\`bash
   curl http://localhost:5000/api/health
   \`\`\`
   Should return: `{"status": "ok", "message": "Backend is running", ...}`

2. **Test image upload:**
   - Go to http://localhost:3000/detect
   - Upload an image
   - Click "Analyze Image"
   - Check browser console and backend terminal for logs

## File Structure

\`\`\`
wheat-disease-detection/
├── app/                     # Next.js frontend
│   ├── detect/             # Disease detection page
│   ├── utils/              # API utilities
│   └── ...
├── backend/                # Flask backend
│   ├── app.py              # Main Flask app
│   ├── setup_dummy_model.py # Creates test model
│   ├── trained_model.keras # Your trained model
│   └── requirements.txt    # Python dependencies
└── package.json            # Node.js dependencies
\`\`\`

## Replacing with Your Model

1. Copy your `trained_model.keras` to the `backend/` directory
2. Update `CLASS_NAMES` in `backend/app.py` if your classes are different
3. Update `DISEASE_MAPPING` in `backend/app.py` for user-friendly names
4. Restart the backend server

## Common Issues

- **Button not working**: Check browser console and backend logs
- **Image not uploading**: Verify file size < 10MB and format is supported
- **Prediction errors**: Check model file exists and TensorFlow is installed
- **Database errors**: Delete `backend/history.db` and restart backend
