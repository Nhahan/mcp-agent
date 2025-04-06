import logging
from pathlib import Path

import httpx
from tqdm import tqdm

logger = logging.getLogger(__name__)

async def download_file_with_progress(url: str, dest_path: Path):
    """Downloads a file from a URL to a destination path with a progress bar."""
    logger.info(f"Downloading file from {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            async with client.stream("GET", url, timeout=None) as response:
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))

                with open(dest_path, "wb") as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=dest_path.name
                ) as progress_bar:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        size = f.write(chunk)
                        progress_bar.update(size)

                if total_size != 0 and progress_bar.n != total_size:
                    logger.error(f"Download incomplete: expected {total_size} bytes, got {progress_bar.n} bytes.")
                    # Consider removing the partial file
                    # dest_path.unlink(missing_ok=True)
                    raise RuntimeError("File download failed: Incomplete file.")
                logger.info(f"File downloaded successfully to {dest_path}.")

        except httpx.RequestError as e:
            logger.error(f"HTTP error occurred during download: {e}")
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to download file from {url}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during download: {e}")
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            raise 