# Authors Steven Ren and Genue Kimm

import torch
import threading
import modelserver_pb2
import modelserver_pb2_grpc
from concurrent import futures
import grpc

class PredictionCache:
    def __init__(self):
        self.lock = threading.Lock()
        self.coefs = None
        self.cache = {}  # LRU cache dictionary
        self.cache_order = []  # List to keep track of LRU cache order
        self.cache_size = 10 # LRU cache size

    def SetCoefs(self, coefs):
        with self.lock:
            self.coefs = coefs
            self.cache.clear()  # Invalidate the cache when SetCoefs is called
            self.cache_order.clear()

    def Predict(self, X):
        with self.lock:
            #if self.coefs is None:
            #    return None, False

            # Round the X values to 4 decimal places
            X = torch.round(X, decimals = 4)

            # When adding an X value to a caching dictionary or looking it up
            # Convert X to a tuple
            X_tuple = tuple(X.flatten().tolist())

            # Check if the prediction is already cached
            if X_tuple in self.cache:
                y = self.cache[X_tuple]
                # Move the key to the end of the cache order list to indicate it was recently used
                self.cache_order.remove(X_tuple)
                self.cache_order.append(X_tuple)
                return y, True
            else:
                # Calculate the prediction
                y = X @ self.coefs

                # Add the prediction to the cache and remove the oldest entry if cache is full
                if len(self.cache) >= self.cache_size:
                    # Remove the least recently used entry
                    lru_key = self.cache_order.pop(0)
                    del self.cache[lru_key]

                self.cache[X_tuple] = y
                self.cache_order.append(X_tuple)
                return y, False
            
class ModelServer(modelserver_pb2_grpc.ModelServerServicer):
    def __init__(self):
        self.prediction_cache = PredictionCache()

    def Predict(self, request, context):
        response = modelserver_pb2.PredictResponse()
        try:
            # Translate the repeated float values from gRPC to a single-row tensor
            X = torch.tensor(request.X, dtype=torch.float32).reshape(1, -1)

            # Calculate the prediction using the cache
            prediction, from_cache = self.prediction_cache.Predict(X)

            # Translate the prediction back to a list of floats
            response.y = prediction[0].item()
            response.hit = from_cache
            response.error = ""  # No error message

        except Exception as e:
            response.error = str(e)  # Set the error message

        return response

    def SetCoefs(self, request, context):
        response = modelserver_pb2.SetCoefsResponse()
        try:
            # Translate the repeated float values from gRPC to a tensor
            coefs = torch.tensor(request.coefs, dtype=torch.float32)

            # Set the coefficients in the cache
            self.prediction_cache.SetCoefs(coefs)
            response.error = ""  # No error message

        except Exception as e:
            response.error = str(e)  # Set the error message

        return response

server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=(('grpc.so_reuseport', 0),))
modelserver_pb2_grpc.add_ModelServerServicer_to_server(ModelServer(), server)
server.add_insecure_port("[::]:5440", ) # trying out 6440 on vim
server.start()
print("started")
server.wait_for_termination()
