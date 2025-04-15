# Modal FastAPI Serverless Server

A scalable, serverless architecture for AI model inference with support for multiple AI services, asynchronous processing, and GCP integration.

## Overview

This repository implements a FastAPI-based server that orchestrates various AI model services using Modal for serverless execution and Celery for asynchronous task processing. The architecture allows for efficient handling of computationally intensive AI tasks, with support for both synchronous and asynchronous processing.

## Features

- **Multiple AI Services**: Support for various AI models and services including:
  - Text-to-Image generation (SDXL, DALLE, Flux, Ideogram, Leonardo)
  - Image-to-Image transformation (inpainting, background removal, upscaling)
  - Video generation
  - Audio generation

- **Serverless Architecture**: Uses Modal for scalable, serverless execution of AI models

- **Asynchronous Processing**: Celery-based task queue for handling long-running tasks for APIs

- **API Authentication**: Secure API access with token-based authentication

- **Cloud Storage Integration**: Support for uploading outputs to Google Cloud Storage

- **Monitoring and Error Handling**: Comprehensive error handling and task status tracking

## API Endpoints

- `/generate_content`: Generate content using specified AI service
- `/list_of_Apps`: List all available AI services
- `/tasks`: Check the status of an asynchronous task
- `/upload_data_GCP`: Upload files to Google Cloud Storage

## Architecture

The system is organized into several components:

- **FastAPI Server**: Handles API requests and routes them to appropriate services
- **Modal Pipelines**: Serverless containers for running AI models on Modal
- **Celery Workers**: Process asynchronous tasks
- **Data Models**: Pydantic models for request/response validation
- **Repositories**: Abstract data access layer
- **Utilities**: Common functions and constants

## Environment Requirements

- Python 3.11.8
- Redis (for Celery task queue)
- Google Cloud Storage account
- Various AI service API keys

## Configuration

The application is configured through environment variables loaded via dotenv. Key configuration includes:

- Google Cloud credentials
- Redis URL for Celery
- API authentication tokens
- Various API service credentials

## Usage

1. Set up environment variables in a `.env` file
2. Start the Redis server for Celery
3. Run the FastAPI server:
   ```
   python main.py
   ```

4. Send requests to the API endpoints with proper authentication
