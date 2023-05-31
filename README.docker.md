# Instructions for Docker

Building the Docker image

```bash
docker build -t bark-infinity:latest .
```

Running the image

```bash
docker run --gpus all -p 7860:7860 -v "$(pwd)/.cache:/root/.cache" -v "$(pwd):/src" --rm -e GRADIO_SERVER_NAME=0.0.0.0 bark-infinity:latest
```

You can now use it at http://localhost:7860
