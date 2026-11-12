import sys
print("Testing SVG save...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2], [3, 4])
fig.savefig("test_fig.svg")
print("Saved SVG successfully!", flush=True)
plt.close(fig)

