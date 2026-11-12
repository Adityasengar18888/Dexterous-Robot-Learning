import sys
print("Hello from test_debug", flush=True)
import pandas as pd
print("Pandas loaded", flush=True)
import matplotlib
print("Matplotlib loaded", flush=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
print("Pyplot loaded", flush=True)
fig, ax = plt.subplots()
print("Subplots created", flush=True)
ax.plot([1, 2], [3, 4])
print("Plotted", flush=True)
fig.savefig("test_fig.png")
print("Saved fig", flush=True)
plt.close(fig)
print("Done!", flush=True)

