"""
Exceptions for the hindcast API and a decorator that turns low-level request
failures into clear, actionable errors.

A 502 or 503 response means the NLR HSDS service is down, which is tracked in
the MHKiT issue below. Any other failure asks the user to open a new issue and
paste the stack trace. To handle a new failure mode, add an exception class and
a branch to :func:`hindcast_guard`.
"""

import functools
import traceback
from typing import Callable

# Existing MHKiT issue tracking the known NLR HSDS service outage.
HINDCAST_UNAVALIABLE_ISSUE_URL = (
    "https://github.com/MHKiT-Software/MHKiT-Python/issues/450"
)
# Where to open a new MHKiT issue for any other hindcast request failure.
NEW_ISSUE_URL = "https://github.com/MHKiT-Software/MHKiT-Python/issues/new"
# HTTP status codes that mean the HSDS service is down (tracked in #450).
_SERVICE_DOWN_CODES = ("502", "503")


class HindcastApiError(RuntimeError):
    """Base class for hindcast API errors, also raised for unexpected failures."""


class HindcastApiUnavailableError(HindcastApiError):
    """Raised when the hindcast API is down (HSDS 502 or 503)."""


def _is_service_down(err: Exception) -> bool:
    """Return True when the error looks like an HSDS 502 or 503 response."""
    text = str(err)
    return any(code in text for code in _SERVICE_DOWN_CODES)


def _unavailable_message(trace: str) -> str:
    return (
        "Could not retrieve the hindcast data. The NLR HSDS endpoint "
        "(developer.nlr.gov) returned a 502 or 503 error and is likely down.\n\n"
        "This is a known outage tracked here:\n"
        f"  {HINDCAST_UNAVALIABLE_ISSUE_URL}\n\n"
        f"Stack trace:\n{trace}"
    )


def _unexpected_message(trace: str) -> str:
    return (
        "An unexpected error occurred while retrieving the hindcast data.\n\n"
        "Please open an issue and paste the stack trace below:\n"
        f"  {NEW_ISSUE_URL}\n\n"
        f"Stack trace:\n{trace}"
    )


def hindcast_guard(func: Callable) -> Callable:
    """
    Convert a failed hindcast API request into a clear HindcastApiError.

    The wrapped function runs normally, so cached results and a recovered
    service still work. Input validation errors (TypeError, ValueError) and
    existing HindcastApiError subclasses pass through unchanged. An HSDS 502
    or 503 raises HindcastApiUnavailableError pointing to the tracking issue.
    Any other failure raises HindcastApiError asking the user to open an
    issue. Both include the stack trace for the user to copy into the issue.

    Parameters
    ----------
    func : callable
        Hindcast data-request function to wrap.

    Returns
    -------
    callable
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError, HindcastApiError):
            raise
        except Exception as err:
            trace = traceback.format_exc()
            if _is_service_down(err):
                raise HindcastApiUnavailableError(_unavailable_message(trace)) from err
            raise HindcastApiError(_unexpected_message(trace)) from err

    return wrapper
