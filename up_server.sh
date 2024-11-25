#!/bin/bash

# Start a Python HTTP server on port 80
python3 -m http.server 80 &

# Run the ConvNetServer executable
./ConvNetServer &

# Keep the script running indefinitely
sleep infinity