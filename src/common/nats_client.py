import asyncio

import nats


async def connect_with_retry(nats_url, service_name,
                             max_attempts=4, base_delay=2.0, max_delay=30.0):
    # one outer attempt may include the client's full initial-connect retry cycle
    # successful connections keep the client's normal reconnect behavior
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            nc = await nats.connect(nats_url)
            if attempt > 1:
                print(f"[{service_name}] connected to nats on attempt {attempt}")
            return nc
        except Exception as e:
            if attempt == max_attempts:
                print(f"[{service_name}] giving up on nats after {max_attempts} attempts")
                raise
            print(f"[{service_name}] nats connect failed (attempt {attempt}/{max_attempts}): {e}, retrying in {delay:.0f}s")
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_delay)
