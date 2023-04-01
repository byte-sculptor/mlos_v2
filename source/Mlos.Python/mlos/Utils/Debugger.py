import debugpy
import traceback


def wait_for_debugger() -> None:

    for port_num in range(5678, 5678 + 12):
        try:
            debugpy.listen(port_num)
            print(f"Waiting for debugger to attach on port {port_num}.", flush=True)
            debugpy.wait_for_client()
            debugpy.breakpoint()
            print("Debugger attached.")
            break
        except RuntimeError as e:
            traceback.print_exc()
            continue

        raise RuntimeError()

    return
