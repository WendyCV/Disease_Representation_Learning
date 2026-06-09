def cleanup_memory():
    import time
    time.sleep(5)
    
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass