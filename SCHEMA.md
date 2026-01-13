# Input / Output Schema

Input:
- Image file path (JPEG/PNG). Preprocessing is deterministic and fixed.

Output (JSON) — `InferenceOutput` (Pydantic):
- `status`: "PREDICT" | "REJECT"
- `cataract_prob`: float | null (0..1)
- `confidence`: float | null (0..1)
- `action`: "PREDICT" | "REFER_TO_SPECIALIST" | "REJECT"
- `reason`: dict | null — structured reason on rejection, e.g. `{ "code": "LOW_BLUR", "metric": 12.3 }`

Schema Version: 1.0
