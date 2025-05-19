import warnings

# Only suppress specific, reviewed warnings here.
# Example: Suppress a known FutureWarning from a specific dependency
# warnings.filterwarnings(
#     "ignore",
#     category=FutureWarning,
#     module=r"^some_dependency\.module$",
#     message=r"This is a known harmless future warning."
# )

# Add more targeted filters as needed, after review.


def configure_warnings():
    """
    Call this function at package import to apply MHKiT's targeted warning filters.
    """
    # Example: Uncomment and edit below to suppress a specific warning
    # warnings.filterwarnings(
    #     "ignore",
    #     category=FutureWarning,
    #     module=r"^some_dependency\.module$",
    #     message=r"This is a known harmless future warning."
    # )
    pass
