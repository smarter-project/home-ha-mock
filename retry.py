from typing import TypeVar, Callable, Optional, Any
import logging
import time

T = TypeVar('T')

def with_retries(
    f: Callable[..., T],
    max_retries: int,
    delay: int,
    description: str = "Operation",
    check_fn: Optional[Callable[[T], bool]] = None
) -> Callable[..., T]:
    """Higher-order function that adds retry capability to any function.
    
    Args:
        f: The function to wrap with retry logic
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        description: Description of the operation for logging
        check_fn: Optional function to check if retry should continue
    """
    def retried_f(*args: Any, **kwargs: Any) -> T:
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                result = f(*args, **kwargs)
                if check_fn is None or check_fn(result):
                    return result
                
                if attempt < max_retries - 1:
                    logging.info(f"{description}: attempt {attempt + 1}/{max_retries} failed, retrying in {delay}s...")
                    time.sleep(delay)
                continue
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logging.info(f"{description}: attempt {attempt + 1}/{max_retries} failed with {str(e)}, retrying in {delay}s...")
                    time.sleep(delay)
                continue
        
        if last_exception:
            raise last_exception
        raise TimeoutError(f"{description} failed after {max_retries} attempts")
    
    return retried_f 