import httpx

async def fetch_json(url: str, timeout: int = 10):
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.status_code, r.text