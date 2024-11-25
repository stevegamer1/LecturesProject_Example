# CUDA Neural Network Web Server with TBB and Docker

This project showcases a complete implementation of a neural network running on CUDA, with TBB-powered image preprocessing, hosted as a web server in Docker containers. It also includes comprehensive testing with Google Test and GitHub Actions for CI/CD.

This repository is a **final project example** for students of the course. For lectures and course materials, visit the [course repository](https://github.com/Liahimz/Lectures).

---

## 🚀 Features

- **CUDA Neural Network**:
  - Fully implemented convolutional neural network leveraging NVIDIA CUDA.
  - Handles both preprocessing and inference for image inputs.

- **TBB-Powered Image Preprocessing**:
  - Efficient parallel image preprocessing using Intel TBB.
  - Supports grayscale conversion, normalization, and Sobel edge detection.

- **Web Server Integration**:
  - Web server interface built using C++ and `cpprestsdk`.
  - Accepts images via HTTP POST requests, processes them, and returns predictions.

- **Dockerized Deployment**:
  - Complete containerized setup with Docker Compose for ease of deployment.
  - Separate Dockerfiles for building and running the server.

- **Testing and CI/CD**:
  - Unit tests implemented with Google Test for image preprocessing.
  - Automated testing on GitHub Actions with full CMake-based builds.

---

## 📂 Project Structure

```plaintext
project-root/
├── data/                   # Network data (weights, biases, etc.)
│   └── network data files
├── docker/                 # Docker setup
│   ├── Dockerfile.builder  # For building the application
│   └── Dockerfile.server   # For running the web server
├── include/                # Header files
│   └── include file(s)
├── src/                    # Source files
│   └── src files
├── utils/                  # Python utilities for local testing
│   └── Python files
├── CMakeLists.txt          # CMake configuration for the project
├── docker-compose.yml      # Docker Compose setup to build and run the server
├── index.html              # Web interface for uploading images
└── README.md               # This file!

## 🛠️ Prerequisites

Before using this project, ensure you have the following installed:

- **CUDA Toolkit** (Version 11 or higher)
- **Docker** and **Docker Compose**
- **CMake** (Version 3.10 or higher)
- **Python 3** (Optional, for local testing utilities)

## 🔧 How to Build and Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository


### 2. Build and Run with Docker

```bash
docker-compose up --build

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and distribute.