#!/bin/bash
# Build the frontend for production

set -e

echo "Building frontend..."

cd "$(dirname "$0")/../frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the frontend
echo "Running build..."
npm run build

echo "Frontend build complete! Output in frontend/dist/"
