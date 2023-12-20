# Authors Steven Ren and Genue Kimm

import sys
import grpc
import threading
import modelserver_pb2
import modelserver_pb2_grpc
import csv

class ClientThread(threading.Thread):
    def __init__(self, port, coefs, csv_file):
        super(ClientThread, self).__init__()
        self.port = port
        self.coefs = coefs
        self.csv_file = csv_file
        self.hits = 0
        self.misses = 0

    def run(self):
        # Connect to the grpc server
        channel = grpc.insecure_channel(f'localhost:{self.port}')
        stub = modelserver_pb2_grpc.ModelServerStub(channel)

        # Set coefficients using SetCoefsRequest
        coefs_list = [float(x) for x in self.coefs.split(',')]
        request = modelserver_pb2.SetCoefsRequest(coefs=coefs_list)
        stub.SetCoefs(request)

        # Read the CSV file call to server
        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convert CSV row to a list of floats
                row_values = [float(val) for val in row if val.strip()]

                # Make a Predict call to the server using PredictRequest
                request = modelserver_pb2.PredictRequest(X=row_values)
                response = stub.Predict(request)

                # Count the hits and misses
                if response.error == "":
                    if response.hit:
                        self.hits += 1
                    else:
                        self.misses += 1
                else:
                    self.misses += 1

def main():
    # Read in command line 
    port = int(sys.argv[1])
    coefs = sys.argv[2]
    csv_files = sys.argv[3:]

    # Create and start threads for each CSV file
    threads = []
    for csv_file in csv_files:
        thread = ClientThread(port, coefs, csv_file)
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Calculate the overall hit rate
    total_hits = sum(thread.hits for thread in threads)
    total_misses = sum(thread.misses for thread in threads)
    overall_hit_rate = total_hits / (total_hits + total_misses)

    # Print the overall hit rate
    print(overall_hit_rate)

if __name__ == '__main__':
    main()
