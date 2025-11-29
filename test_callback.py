import cupy as cp


# 1. Define the callback function
def my_callback(stream, status, user_data):
    # This function is executed on the CPU by a driver thread.
    # It cannot call any CUDA API functions.
    print(f"Callback received for stream: {user_data['name']}")
    print(f"Error status: {status}")

# Create a non-default stream
s = cp.cuda.Stream(non_blocking=True)

# Create some dummy arrays
a = cp.ones(10, dtype=cp.float32)
b = cp.ones(10, dtype=cp.float32)

# Perform some asynchronous operations on the stream
with s:
    c = a + b
    d = c * a

# 2. Add the callback to the stream's command queue
# The callback function will run on the CPU after all the operations above are complete.
user_info = {'name': 'my_custom_stream'}
s.add_callback(my_callback, user_info)

# The following line will block the main thread and ensure the callback is executed.
# This explicitly mimics waiting for the stream to complete.
print("Main thread waiting for stream to finish...")
s.synchronize()

print("Main thread finished.")
