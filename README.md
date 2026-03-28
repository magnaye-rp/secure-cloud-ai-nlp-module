# Secure Cloud AI NLP Module

Production-ready NLP AI service deployed on Kubernetes with **CIA triad** best practices.

## Features
- FastAPI endpoint for NLP (sentiment analysis starter)
- Containerized with Docker
- Kubernetes manifests (Deployment + HPA + probes)
- GitOps-ready structure
- Built-in rate limiting, secrets, and security headers

## Quick Start (Local)
```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
