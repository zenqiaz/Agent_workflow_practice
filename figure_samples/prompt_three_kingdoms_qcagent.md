# AI Image Generator Prompt — Three Kingdoms QC Agent Overview Figure

---

## Prompt

A scientific diagram illustration in a playful "emoji-robot" style, depicting a Three Kingdoms command hierarchy as a metaphor for an AI quantum-chemistry agent system. The user is a real human figure in Chinese emperor costume; the planner and orchestrator are chibi-robot figures with round metallic heads, large LED eyes, simple cylindrical bodies, and small hands — all dressed in Three Kingdoms Chinese costume as described below. Software logos appear as flat icon badges reproduced exactly from the uploaded reference images. The composition is clean on a plain white background with no decorative background elements. Minimal English text labels in small sans-serif font for arrows and panels only — do NOT label characters with their names.

---

### Layout Description

**Two-panel composition separated by a horizontal boundary line:**

```
┌─────────────────────────────────────────────────────┐
│  Client                                             │
│                                                     │
│            User (human)                             │
│          ↙            ↘                             │
│    Planner          Orchestrator                    │
│   (robot)             (robot)                       │
│                          ↓                          │
├ ─ ─ ─ ─ ─ ─ MCP over SSH ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
│                          ↓                          │
│  Server                                             │
│                                                     │
│   [ORCA]  [xTB]  [PubChem]  [molSimplify]  [RDKit]  │
└─────────────────────────────────────────────────────┘
```

---

### Panel 1 — Client (upper half, light blue background panel)

The upper half of the figure sits inside a rounded rectangle with a very light blue fill (#EEF4FB) and a solid blue border, labeled **"Client"** in the top-left corner in small bold text. No platform name (no "Windows", no "local").

### Zone 1 — The Triangle of Command (upper portion)

**User — top center:**
- A **real human figure** (not a robot), drawn in a clean cartoon/chibi style consistent with the other characters
- Wearing a yellow dragon emperor robe (龙袍) with gold embroidery borders and a tall imperial crown
- Friendly expression, left hand raised slightly as if issuing an order
- **No name label.** Arrow labeled **"request"** flows downward-left toward the Planner; arrow labeled **"report"** returns from the Orchestrator

**Planner — lower left:**
- Chibi robot figure
- Wearing a white Taoist scholar robe with wide sleeves and a tall black cloth cap (葛巾)
- Right hand holding a large white feather fan (羽扇) with subtle circuit-board trace patterns
- Left hand holding a glowing golden silk drawstring pouch — the plan artifact. The pouch has small text tags floating near it reading **"nodes"** and **"artifacts"** (as if the contents of the bag are labeling themselves). Do NOT write "jinliang" or any Chinese text on or near the bag.
- Beside the Planner (to their left or at their feet) stands a small neat stack of **ancient Chinese books** — rectangular hardcover books in the style of classical Chinese thread-bound volumes (线装书), spines facing outward. Each book spine has a short label in small English text representing a skill module: **"pKa"**, **"method"**, **"solv"**, **"thermo"**, **"NBO"**. The books are slightly different heights, stacked upright like reference volumes on a shelf. They have a warm tan/ivory color with red or dark thread binding detail.
- LED eyes show a thoughtful expression
- **No name label.** Arrow labeled **"plan"** (golden, solid) flows rightward toward the Orchestrator, with a small pouch icon on the arrow shaft

**Orchestrator — lower right:**
- Chibi robot figure, slightly larger than the Planner
- Wearing a green battle robe (青袍) over bronze scale armor, long stylized black beard, warm red-tinted faceplate
- Holding a halberd upright
- LED eyes show a determined expression
- Positioned at the bottom edge of the client panel, nearest the MCP boundary line
- **No name label.**

**Connecting arrows in the triangle:**
- User → Planner: dashed curved arrow, labeled **"request"**, orange
- Planner → Orchestrator: solid curved arrow, labeled **"plan"**, golden, small pouch icon on arrow
- Orchestrator → User: dashed return arrow, labeled **"report"**, teal
- Planner ↔ Orchestrator: thin dashed bidirectional line labeled **"state"**, grey

---

### Boundary — MCP Communication Line

A visually prominent horizontal divider: thick dashed line in deep teal (#2A7A6F). Centered on it: a rectangular badge reading **"MCP over SSH"** in bold. Two bold teal arrows pierce the boundary — ↓ labeled **"tool calls"** and ↑ labeled **"results"** — making cross-boundary communication the most visually salient element.

### Panel 2 — Server (lower half, light green background panel)

The lower half sits inside a rounded rectangle with a very light green fill (#EEF8F1) and a solid green border, labeled **"Server (k8s / SLURM)"** in the top-left corner. No "remote" qualifier.

---

### Zone 2 — The Soldier Row (server panel)

Five chibi robot soldiers standing in a neat horizontal line: standing at attention, left hand holding a small shield, right hand holding a glowing test-tube. Simple leather scale armor (鱼鳞甲) with bronze helmet. LED eyes show "O_O" focused expression.

Each soldier has a software logo badge on their shield, reproduced **exactly** from the uploaded reference images — do not substitute, simplify, or hallucinate any logo. You can ONLY slightly adjust the relative SIZE of logos, and it is ok if the widths of soldiers differ due to the size of logo, but not too much. Text label below each soldier's feet:

| Position | Shield Logo | Label |
|----------|------------|-------|
| 1 | ORCA logo (uploaded) | **"ORCA"** |
| 2 | xTB logo (uploaded) | **"xTB"** |
| 3 | PubChem logo (uploaded) | **"PubChem"** |
| 4 | molSimplify logo (uploaded) | **"molSimplify"** |
| 5 | RDKit logo (uploaded) | **"RDKit"** |

The MCP boundary is the single crossing point for all tool calls and results — no direct arrows from the Orchestrator to individual soldiers.

---

### Decorative Elements

- **Background:** Plain white. No bamboo, clouds, stamps, or border ornaments.
- **Color palette:** Imperial gold (#D4AF37) for Planner accents and plan arrow, jade green (#4E8B6F) for Orchestrator robe, yellow (#E8C040) for User robe, ink black (#1A1A2E) for outlines and labels. Client panel: light blue (#EEF4FB) fill, blue border. Server panel: light green (#EEF8F1) fill, green border. MCP boundary: deep teal (#2A7A6F) dashed line and arrows. Flat colors, minimal shading.
- **Overall mood:** Clean academic diagram; characters add visual interest but must not distract from the information hierarchy.

---

### Style Keywords (for diffusion model)

`chibi robot characters, one human character, ancient Chinese costume, Three Kingdoms aesthetic, emoji expression faces, LED eyes, flat design, clean white background, labeled diagram, hierarchy chart, molecular software logos faithfully reproduced, arrows and flow indicators, academic figure style, minimal decoration, no character name labels`

---

### Negative Prompt

`photorealistic, dark background, horror, violence, modern clothing, anime shonen style, busy cluttered background, bamboo, clouds, seal stamps, border ornaments, ink wash texture, parchment, character name text labels, "jinliang", "Liu Bei", "Zhuge Liang", "Guan Yu", "Windows", "Remote", distorted hands, floating limbs, NSFW, wrong logos, generic placeholder logos`
