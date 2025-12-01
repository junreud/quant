
import json
import nbformat
from pathlib import Path

notebook_path = Path("notebooks/04_model_performance_analysis.ipynb")

if not notebook_path.exists():
    print(f"❌ Notebook not found: {notebook_path}")
    exit(1)

print(f"📖 Reading {notebook_path}...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Inject Setup Cell at the top
setup_code = [
    "# Auto-Save Plots to Experiment Folder\n",
    "import os\n",
    "from pathlib import Path\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Find latest experiment folder\n",
    "project_root = Path('..') # Assuming notebook is in notebooks/\n",
    "exp_dir = project_root / 'experiments'\n",
    "if exp_dir.exists():\n",
    "    experiments = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith('exp_')], key=lambda x: x.name)\n",
    "    if experiments:\n",
    "        LATEST_EXP_PATH = experiments[-1]\n",
    "        print(f\"📂 Saving plots to: {LATEST_EXP_PATH}\")\n",
    "    else:\n",
    "        LATEST_EXP_PATH = None\n",
    "        print(\"⚠️ No experiment folder found.\")\n",
    "else:\n",
    "    LATEST_EXP_PATH = None\n",
    "\n",
    "def save_plot(filename):\n",
    "    if LATEST_EXP_PATH:\n",
    "        path = LATEST_EXP_PATH / filename\n",
    "        plt.savefig(path, bbox_inches='tight')\n",
    "        print(f\"💾 Saved: {path.name}\")\n"
]

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code
}

# Insert setup cell at index 1 (after imports usually)
nb['cells'].insert(1, setup_cell)

# 2. Inject save_plot calls
# Iterate through cells and check for plotting commands
plot_keywords = ['plt.show()', 'plt.plot', 'sns.heatmap', 'sns.barplot', 'plot(kind=']
count = 0

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if any(kw in source for kw in plot_keywords):
            # Check if already has save_plot to avoid duplicates
            if "save_plot(" in source:
                continue
                
            # Append save_plot call
            # Generate a unique filename based on cell index or content hint
            filename = f"plot_cell_{i}.png"
            
            # Add before plt.show() if it exists, otherwise at end
            lines = cell['source']
            new_lines = []
            saved = False
            
            for line in lines:
                if "plt.show()" in line and not saved:
                    new_lines.append(f"save_plot('{filename}')\n")
                    saved = True
                new_lines.append(line)
            
            if not saved:
                new_lines.append(f"\nsave_plot('{filename}')")
                
            cell['source'] = new_lines
            count += 1

print(f"✅ Injected save_plot to {count} cells.")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("🎉 Notebook updated with auto-save logic.")
