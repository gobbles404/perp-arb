from binance_client import client

# Get all available methods
spot_methods = [method for method in dir(client) if callable(getattr(client, method))]

# Print the list
for method in spot_methods:
    print(method)
