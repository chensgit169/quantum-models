import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(figsize=(6,6))

# Axis limits
x_min, x_max = 0, 3
y_min, y_max = 0, 3

# Background = remaining region (first quadrant)
bg = Rectangle((0, 0), x_max, y_max,
               facecolor='none', edgecolor='none', zorder=0)
ax.add_patch(bg)

# Region A: 0 < x < 1 (vertical strip) — forward-slash hatch
rect_A = Rectangle((0, 0), 1, y_max,
                   facecolor='none', edgecolor='black',
                   hatch='///', linewidth=0.8, zorder=2)
ax.add_patch(rect_A)

# Region B: 0 < y < 1 (horizontal strip) — cross (x) hatch
rect_B = Rectangle((0, 0), x_max, 1,
                   facecolor='none', edgecolor='black',
                   hatch='\\\\\\', linewidth=0.8, zorder=3)
ax.add_patch(rect_B)

# Boundary lines for x=1 and y=1
ax.axvline(1, color='k', linestyle='--', zorder=5)
ax.axhline(1, color='k', linestyle='--', zorder=5)

# === Keep only ticks 0 and 1 ===
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

# Hide top and right borders
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# Axes settings
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.set_xlabel(r'$A$', fontsize=14)
ax.set_ylabel(r'$\Delta$', fontsize=14)
# ax.set_title('Three regions in the first quadrant with hatched filling')

# Legend
legend_patches = [
    Patch(facecolor='none', edgecolor='black', hatch='///', label=r'Region $C_1$: 0 < $A$ < 1'),
    Patch(facecolor='none', edgecolor='black', hatch='\\\\\\', label=r'Region $C_2$: 0 < $\Delta$ < 1'),
    Patch(facecolor='none', edgecolor='black', label=r'Region $C_3$: $A$ > 1, $\Delta$ > 1'),
    Patch(facecolor='none', edgecolor='black', hatch='xxx', label=r'Region $C_0$=$C_1$$\cap$$C_2$'),
]

# === Text labels with white outline ===
outline = [pe.Stroke(linewidth=6, foreground='white'), pe.Normal()]
ax.text(0.5, 0.5, r'$C_0$', ha='center', va='center', fontsize=16, fontweight='bold', path_effects=outline)
ax.text(0.5, 1.5, r'$C_1$', ha='center', va='center', fontsize=16, fontweight='bold', path_effects=outline)
ax.text(1.5, 0.5, r'$C_2$', ha='center', va='center', fontsize=16, fontweight='bold', path_effects=outline)
ax.text(1.5, 1.5, r'$C_3$', ha='center', va='center', fontsize=16, fontweight='bold', path_effects=outline)

ax.legend(handles=legend_patches, loc='upper right', fontsize=14)
plt.tight_layout()
plt.savefig('figures/limiting/param_regions.pdf', dpi=400)
# plt.show()
