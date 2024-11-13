# src/core/exceptions.py
class InfrastructureAnalyzerError(Exception):
    """Base exception for Infrastructure Analyzer."""
    pass

class ClientConnectionError(InfrastructureAnalyzerError):
    """Exception raised for client connection errors."""
    pass

class ConfigurationError(InfrastructureAnalyzerError):
    """Exception raised for configuration errors."""
    pass

class ModelError(InfrastructureAnalyzerError):
    """Exception raised for model-related errors."""
    pass