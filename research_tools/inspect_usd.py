from pxr import Usd

usd_path = "/home/tempest/code/NavDP/assets/robots/lekiwi/lekiwi.usd"
try:
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"Failed to open stage at {usd_path}")
    else:
        print(f"Successfully opened stage: {usd_path}")
        print("Root Layer primitive structure:")
        for prim in stage.Traverse():
            print(prim.GetPath())
except Exception as e:
    print(f"Error opening USD: {e}")
