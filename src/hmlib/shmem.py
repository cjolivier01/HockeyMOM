import numpy as np
from multiprocessing import Process, shared_memory
import time

def consume_image(shm_name):
    # Attach to the shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)

    # Create a NumPy array backed by shared memory
    image_shape = (480, 640, 3)  # For example, 480x640 RGB image
    np_array = np.ndarray(image_shape, dtype=np.uint8, buffer=existing_shm.buf)

    print(np_array)
    existing_shm.close()

if __name__ == "__main__":
    # Create a sample image using NumPy
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Create shared memory block
    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)

    # Copy NumPy array to shared memory
    shm.buf[:image.nbytes] = image.tobytes()

    # Start a new process to consume the image
    p = Process(target=consume_image, args=(shm.name,))
    p.start()
    p.join()

    # Clean up
    shm.unlink()
