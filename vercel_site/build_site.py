"""Copy the latest local project metrics and graphs into the Vercel static site."""

import json
import shutil
from pathlib import Path


SITE = Path(__file__).resolve().parent
OUTPUTS = SITE.parent
FIGURES = SITE / "figures"
FIGURES.mkdir(exist_ok=True)

data = {
    "metrics": json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8")),
    "forecast": json.loads((OUTPUTS / "forecast.json").read_text(encoding="utf-8")),
}
(SITE / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
for filename in ["price_indicators.png", "test_predictions.png", "training_history.png"]:
    shutil.copy2(OUTPUTS / filename, FIGURES / filename)
print(f"Vercel static site is ready: {SITE}")
