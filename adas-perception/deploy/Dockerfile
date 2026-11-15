# Forward-perception edge module for Azure IoT Edge on an ADAS ECU.
# Runs the ONNX detector on CPU inside an AMD SEV-SNP confidential VM.
FROM mcr.microsoft.com/azureml/onnxruntime:latest-cpu

WORKDIR /opt/perception

# Lightweight runtime only — no torch / ultralytics on the ECU.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models/ ./models/
COPY src/ ./src/
COPY data/ ./data/
COPY deploy/ecu_config.yaml ./ecu_config.yaml

# Health/inference entrypoint (single-frame demo; the module host loops frames).
ENV PERCEPTION_MODEL=/opt/perception/models/tiny-yolov3-11.onnx
ENTRYPOINT ["python", "src/perception.py"]
