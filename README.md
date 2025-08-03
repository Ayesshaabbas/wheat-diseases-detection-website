# WheatGuard - Wheat Disease Detection System

This project combines a Next.js frontend with a Flask backend to create a wheat disease detection system.

## Step-by-Step Setup Instructions for Windows

### Prerequisites

1. **Install Node.js**
   - Download from https://nodejs.org/
   - Choose the LTS version (recommended)
   - Run the installer and follow the setup wizard
   - Verify installation by opening Command Prompt and typing: `node --version`

2. **Install Python**
   - Download from https://www.python.org/downloads/
   - Choose Python 3.8 or later
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Verify installation by opening Command Prompt and typing: `python --version`

### Installation Steps

#### Step 1: Extract and Navigate to Project
1. Extract the downloaded zip file to your desired location (e.g., `C:\Projects\wheat-disease-detection`)
2. Open Command Prompt (cmd) or PowerShell
3. Navigate to the project directory:
   ```cmd
   cd C:\Projects\wheat-disease-detection
   \`\`\`

#### Step 2: Install Frontend Dependencies
1. In the project root directory, run:
   ```cmd
   npm install
   \`\`\`
   This will install all the required Node.js packages.

#### Step 3: Setup Backend
1. Navigate to the backend directory:
   ```cmd
   cd backend
   \`\`\`

2. Create a Python virtual environment:
   ```cmd
   python -m venv venv
   \`\`\`

3. Activate the virtual environment:
   ```cmd
   venv\Scripts\activate
   \`\`\`
   You should see `(venv)` at the beginning of your command prompt.

4. Install Python dependencies:
   ```cmd
   pip install -r requirements.txt
   \`\`\`

#### Step 4: Running the Application

You need to run both the frontend and backend servers simultaneously.

**Option 1: Using Two Command Prompt Windows**

1. **Terminal 1 - Backend Server:**
   ```cmd
   cd C:\Projects\wheat-disease-detection\backend
   venv\Scripts\activate
   python app.py
   \`\`\`
   You should see: "Running on http://0.0.0.0:5000"

2. **Terminal 2 - Frontend Server:**
   ```cmd
   cd C:\Projects\wheat-disease-detection
   npm run dev
   \`\`\`
   You should see: "Ready - started server on 0.0.0.0:3000"

**Option 2: Using PowerShell (Recommended)**

1. Open PowerShell as Administrator
2. Navigate to your project directory
3. Run the backend:
   ```powershell
   Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; .\venv\Scripts\Activate.ps1; python app.py"
   \`\`\`
4. Run the frontend:
   ```powershell
   npm run dev
   \`\`\`

### Accessing the Application

1. **Frontend**: Open your web browser and go to `http://localhost:3000`
2. **Backend API**: The Flask API runs on `http://localhost:5000`

### Using the Application

1. **Sign Up/Login**: Create an account or log in
2. **Detect Disease**: 
   - Go to "Detect Disease" page
   - Upload an image of wheat crop
   - Wait for AI analysis
   - View results
3. **View History**: Check your past detection results on the History page

### Troubleshooting Backend Connection Issues

If you're experiencing backend connection issues:

1. **Verify the backend is running:**
   - Make sure you see "Running on http://0.0.0.0:5000" in your terminal
   - Try accessing http://localhost:5000/api/test in your browser - you should see a JSON response

2. **Check for port conflicts:**
   - If port 5000 is already in use, you'll need to change it in `backend/app.py`
   - If you change the port, also update the `BACKEND_URL` in `app/utils/api.ts`

3. **Check CORS settings:**
   - The backend is configured to accept requests from any origin
   - If you're still having CORS issues, check your browser console for specific errors

4. **Firewall issues:**
   - Make sure your firewall isn't blocking connections to localhost:5000
   - Try temporarily disabling your firewall to test

5. **Database issues:**
   - If you see database errors, delete the `backend/history.db` file and restart the backend
   - The database will be recreated automatically

### General Troubleshooting

**If you get "python is not recognized":**
- Reinstall Python and make sure to check "Add Python to PATH"
- Or manually add Python to your PATH environment variable

**If you get "npm is not recognized":**
- Reinstall Node.js
- Restart your command prompt after installation

**If the backend doesn't start:**
- Make sure you're in the backend directory
- Make sure the virtual environment is activated
- Check if port 5000 is already in use

**If the frontend doesn't connect to backend:**
- Make sure both servers are running
- Check that the backend is running on port 5000
- Check your firewall settings

### Project Structure

\`\`\`
wheat-disease-detection/
├── app/                    # Next.js frontend
│   ├── api/                # Next.js API routes
│   ├── detect/             # Disease detection page
│   ├── history/            # History page
│   ├── results/            # Results page
│   ├── utils/              # Utility functions
│   └── ...                 # Other Next.js pages
├── backend/                # Flask backend
│   ├── app.py              # Main Flask application
│   ├── requirements.txt    # Python dependencies
│   └── history.db          # SQLite database (created automatically)
├── package.json            # Node.js dependencies
└── README.md               # This file
\`\`\`

### Features

- **User Authentication**: Simple login/signup system
- **Disease Detection**: AI-powered wheat disease identification
- **History Tracking**: View past detection results
- **Responsive Design**: Works on desktop and mobile devices
- **Offline Mode**: Application works even when backend is unavailable

### Development Notes

- The database (SQLite) is created automatically when you first run the Flask application
- Sample images are provided for testing the disease detection feature
- The AI model is currently a mock implementation for demonstration purposes
