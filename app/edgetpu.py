import logging
import os

logger = logging.getLogger(__name__)

_edgetpu_available = None


def is_edgetpu_available() -> bool:
    """Check if a Coral Edge TPU device is accessible."""
    global _edgetpu_available
    if _edgetpu_available is not None:
        return _edgetpu_available

    # Check for USB or PCIe Coral
    usb_exists = os.path.exists("/dev/bus/usb")
    apex_exists = os.path.exists("/dev/apex_0")

    if not (usb_exists or apex_exists):
        logger.info("No Coral TPU device path found")
        _edgetpu_available = False
        return False

    try:
        import tflite_runtime.interpreter as tflite
        tflite.load_delegate("libedgetpu.so.1")
        _edgetpu_available = True
        logger.info("Coral Edge TPU delegate available")
        return True
    except (ImportError, ValueError, RuntimeError) as e:
        logger.info("Coral Edge TPU delegate not available: %s", e)
        _edgetpu_available = False
        return False


def create_interpreter(cpu_model_path: str, edgetpu_model_path: str):
    """Create a TFLite interpreter, preferring Edge TPU if available.

    Args:
        cpu_model_path: Path to the standard TFLite model.
        edgetpu_model_path: Path to the Edge TPU compiled model.

    Returns:
        (interpreter, using_edgetpu) tuple.
    """
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        logger.warning("tflite_runtime not installed, trying tensorflow.lite")
        import tensorflow.lite as tflite

    if is_edgetpu_available() and os.path.exists(edgetpu_model_path):
        try:
            delegate = tflite.load_delegate("libedgetpu.so.1")
            interpreter = tflite.Interpreter(
                model_path=edgetpu_model_path,
                experimental_delegates=[delegate],
            )
            interpreter.allocate_tensors()
            logger.info("Loaded model on Edge TPU: %s", os.path.basename(edgetpu_model_path))
            return interpreter, True
        except (ValueError, RuntimeError) as e:
            logger.warning(
                "Failed to load Edge TPU model %s: %s. Falling back to CPU.",
                edgetpu_model_path, e,
            )

    if not os.path.exists(cpu_model_path):
        raise FileNotFoundError(f"Model not found: {cpu_model_path}")

    interpreter = tflite.Interpreter(model_path=cpu_model_path)
    interpreter.allocate_tensors()
    logger.info("Loaded model on CPU: %s", os.path.basename(cpu_model_path))
    return interpreter, False
