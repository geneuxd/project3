FROM ubuntu:22.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch --no-cache-dir
RUN pip install numpy 
RUN pip3 install grpcio==1.58.0 grpcio-tools==1.58.0
COPY . .
RUN python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. modelserver.proto
CMD ["python3", "/server.py"]
