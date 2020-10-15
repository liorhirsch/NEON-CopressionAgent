from datetime import datetime

def print_flush(msg):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"{dt_string} -- {msg}", flush=True)

