const { spawn } = require("child_process")
const path = require("path")

// Start Next.js frontend
const frontend = spawn("npm", ["run", "dev"], {
  stdio: "inherit",
  shell: true,
})

// Start Flask backend
const backend = spawn("python", ["backend/app.py"], {
  stdio: "inherit",
  shell: true,
})

// Handle process termination
process.on("SIGINT", () => {
  frontend.kill("SIGINT")
  backend.kill("SIGINT")
  process.exit()
})

console.log("Development servers started:")
console.log("- Frontend: http://localhost:3000")
console.log("- Backend: http://localhost:5000")
