#!/bin/bash

# Create necessary directories
mkdir -p temp data assets

# Start Gunicorn server
gunicorn app:server
