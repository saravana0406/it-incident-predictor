import random
import time
import requests

while True:
    cpu = random.randint(10, 95)
    memory = random.randint(20, 95)
    errors = random.randint(0, 10)

    res = requests.get(
        "http://127.0.0.1:8000/predict",
        params={"cpu": cpu, "memory": memory, "errors": errors}
    )

    print(cpu, memory, errors, res.json())

    time.sleep(2)