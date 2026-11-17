# UI/UX Design Specification: Football Match Ball Possession Prediction Dashboard

## 1. Agent Mission & Figcraft Execution Context
**Objective:** Create a high-craft, clean, modern Bento-grid UI/UX dashboard mockup inside **Figcraft** (via Figcraft MCP tools and CLI) for a real-time **Football Match Ball Possession Prediction** system.
**Scope:** UI/UX design generation in Figcraft **ONLY**. Focus on exact layout structure, design tokens, vector data visualizations, and component hierarchy. Do not generate frontend/backend application code in this phase.

### Reference Alignment
Use the provided reference dashboard layout (`1440px x 1024px` desktop viewport):
- **Aesthetic:** Soft-minimalist, airy light-mode analytics interface with high-contrast charcoal typography, generous negative space, pill-shaped interactive controls, and rounded Bento cards.
- **Domain Adaptation:** Map every financial/regional widget from the reference directly into real-time football telemetry and ball-possession prediction metrics (Home Team vs. Away Team vs. Contested/Loose Ball).

---

## 2. Design Tokens & Visual System

### 2.1 Color Palette
| Token Name | Hex Value | Usage |
| :--- | :--- | :--- |
| `bg-canvas` | `#F6F7F9` | Main dashboard background canvas |
| `bg-surface` | `#FFFFFF` | Bento card surfaces, floating tooltips, inactive icon pills |
| `ink-primary` | `#111315` | Primary headings, large KPI metrics, active nav pill, active badge fill |
| `ink-secondary` | `#6E7684` | Subtitles, metric descriptions, axis labels |
| `ink-muted` | `#9CA3AF` | Dotted divider lines, inactive states, secondary captions |
| `accent-blue` | `#1D6FF2` | **Home Team** possession, primary active curve, dominant pitch zones |
| `accent-blue-mid` | `#5B9BF8` | Secondary streamgraph layer (Home momentum) |
| `accent-blue-light` | `#9CC3FC` | Outer streamgraph layer (Home momentum) |
| `accent-green` | `#38C976` | **Away Team** possession, secondary spline curve, moderate pitch zones |
| `accent-amber` | `#F5B749` | **Contested / Loose Ball** state, third spline curve, low-control zones |
| `accent-neutral` | `#D1D5DB` | Out of play / Set-piece / Neutral pitch zones |
| `border-subtle` | `#ECEEF2` | Subtle card borders and chart gridlines |

### 2.2 Typography (`Plus Jakarta Sans` or `Inter`)
- **Page Title (`Dashboard` / `Match Possession`):** `32px` (`2rem`), SemiBold (`600`), Tracking `-0.02em`, Color `ink-primary`.
- **Top Header KPI Numbers:** `30px` (`1.875rem`), Bold (`700`), Tracking `-0.02em`, Color `ink-primary`.
- **Card Section Titles:** `15px` (`0.9375rem`), SemiBold (`600`), Color `ink-primary`.
- **Center Gauge Hero Metric:** `32px` (`2rem`), Bold (`700`), Color `ink-primary`.
- **Card Sub-KPI Numbers:** `20px` (`1.25rem`), Bold (`700`), Color `ink-primary`.
- **Body / Legend Labels:** `13px` (`0.8125rem`), Medium (`500`), Color `ink-primary` / `ink-secondary`.
- **Micro Labels / Axis / Badges:** `11px` (`0.6875rem`), SemiBold (`600`).

### 2.3 Radii & Elevation
- **Outer Shell Frame:** `32px` border radius, `1440px` width × `1024px` height, fill `#F6F7F9`.
- **Bento Cards:** `24px` border radius, fill `#FFFFFF`, subtle shadow `0px 4px 24px rgba(17, 19, 21, 0.03)`.
- **Circular Icon Buttons & Nav Items:** `44px x 44px` (`border-radius: 9999px`).
- **Floating Pitch Overlay Card:** `18px` border radius, fill `#FFFFFF`, shadow `0px 10px 30px rgba(17, 19, 21, 0.08)`.
- **Chart Pill Badges:** `9999px` border radius, height `26px`, horizontal padding `12px`.

---

## 3. Structural Layout & Grid Architecture

Create a root Auto-Layout frame named `Football_Possession_Dashboard` (`1440px x 1024px`, Horizontal layout, padding `28px`, gap `24px`):

1. **Left Navigation Rail (`width: 64px`, Vertical Auto-Layout, `justify-content: space-between`):**
   - **Top Group:**
     - Brand Logo Mark (`44px x 44px` container): Stylized 3-bar dynamic speed/ball-trajectory icon in `#111315`.
     - Vertical Stack (`gap: 12px`, `margin-top: 32px`) of 7 circular icon buttons (`44px x 44px`):
       1. `Live Match Overview` (**Active State**: Fill `#111315`, White 4-square grid icon)
       2. `Tactical Pitch View` (Inactive: Fill `#FFFFFF`, Dark pitch/map icon)
       3. `Model Telemetry` (Inactive: Fill `#FFFFFF`, Dark clipboard/metrics icon)
       4. `Pass Vectors` (Inactive: Fill `#FFFFFF`, Dark navigation/arrow icon)
       5. `Match Timeline` (Inactive: Fill `#FFFFFF`, Dark clock icon)
       6. `AI Commentary / Alerts` (Inactive: Fill `#FFFFFF`, Dark headset/signal icon)
       7. `Model Settings` (Inactive: Fill `#FFFFFF`, Dark gear icon)
   - **Bottom Group (`gap: 12px`):**
     - Circular Analyst Profile Avatar (`44px x 44px`)
     - Circular Exit / Reset Session Button (`44px x 44px`, Fill `#FFFFFF`)

2. **Main Content Area (`flex: 1`, Vertical Auto-Layout, `gap: 24px`):**
   - **Top Header Bar (`height: 64px`, Horizontal Auto-Layout, `justify-content: space-between`, `align-items: center`):**
     - **Left:** Page Title `"Possession AI"` (`32px`, Bold) + subtle live indicator pill `"LIVE • 67:14'"` (`#1D6FF2` text on `#EBF3FE` pill).
     - **Right (`gap: 40px`, Horizontal Auto-Layout):**
       - **KPI Block 1:** `"14.8K"` (`30px` Bold) next to 2-line label `"Tracking frames\nanalyzed in real-time"` (`12px` `#6E7684`).
       - **KPI Block 2:** `"91%"` (`30px` Bold) next to 2-line label `"Possession prediction\nmodel confidence"` (`12px` `#6E7684`).
   - **Bento Grid Container (`flex: 1`, 2 Columns × 2 Rows, `gap: 20px`):**
     - Row 1: `Card_Current_Possession` (Left, 50%) + `Card_Pitch_Zone_Control` (Right, 50%)
     - Row 2: `Card_Possession_Momentum` (Left, 50%) + `Card_Probability_Trend` (Right, 50%)

---

## 4. Detailed Bento Card Specifications

### Card 1 (Top-Left): `Card_Current_Possession`
*Replaces "Borrowers by State" in reference.*
- **Card Header:**
  - Title: `"Predicted Ball Control"` (`15px` SemiBold)
  - Right Actions (`gap: 8px`): Circular Match-Half Filter button (`36px`, `#F6F7F9` fill, calendar/clock icon) + Circular More Options button (`36px`, `#F6F7F9` fill, vertical 3-dots icon).
- **Card Body (Horizontal Split: 50% Gauge / 50% Breakdown List):**
  - **Left — 240° Segmented Arc Gauge:**
    - Thick rounded-cap arc (`stroke-width: 16px`, `radius: 95px`).
    - Segment 1 (Right/Top-Right, ~64% of arc): `#1D6FF2` (Home Team).
    - Segment 2 (Top-Left/Left, ~27% of arc): `#38C976` (Away Team).
    - Segment 3 (Bottom-Left, ~9% of arc): `#F5B749` (Contested).
    - Center Stack: `"64.2%"` (`32px` Bold `#111315`) above `"Home Possession"` (`12px` `#6E7684`).
  - **Right — Team & State Legend List (Vertical Stack, `gap: 18px`):**
    - Row 1: Blue dot (`10px`) + `"MCI (Home)"` + dotted horizontal line + `"64.2%"` (Bold)
    - Row 2: Green dot (`10px`) + `"ARS (Away)"` + dotted horizontal line + `"26.5%"` (Bold)
    - Row 3: Amber dot (`10px`) + `"Contested"` + dotted horizontal line + `"6.8%"` (Bold)
    - Row 4: Gray dot (`10px`) + `"Out of Play"` + dotted horizontal line + `"2.5%"` (Bold)

### Card 2 (Top-Right): `Card_Pitch_Zone_Control`
*Replaces "Map Preview" in reference.*
- **Card Header:**
  - Title: `"Pitch Zone Dominance"` (`15px` SemiBold)
  - Right Actions (`gap: 8px`): Circular Formation/Zone Filter button (`36px`, `#F6F7F9`) + Circular More Options button (`36px`, `#F6F7F9`).
- **Card Body (Interactive Tactical Pitch Map):**
  - **Center Visualization:** A clean, flat-vector top-down **Football Pitch** (with subtle white pitch markings: center circle, penalty boxes, halfway line) divided into **6 distinct tactical blocks/polygons** (Left/Center/Right zones across Defensive, Midfield, and Attacking thirds), styled identically to the color-blocked regional map in the reference:
    - Central Midfield & Left Wing blocks: `#1D6FF2` (Home dominant)
    - Right Wing & Defensive Third blocks: `#F5B749` (Contested / Transition)
    - Opposition Penalty Area block: `#38C976` (Away control)
    - Deep Corner zones: `#D1D5DB` (Neutral)
  - **Floating Center Callout Card:** Centered over the pitch (`width: 190px`, `padding: 16px 20px`, `#FFFFFF` fill, `18px` radius, soft elevation shadow):
    - Top Text: `"71.8%"` (`28px` Bold `#111315`)
    - Bottom Text: `"Central Midfield"` (`12px` `#6E7684`)
  - **Bottom-Left Map Controls (Vertical Stack, `gap: 8px`):**
    - White circle (`36px`): `+` icon
    - White circle (`36px`): `-` icon
    - Solid Black circle (`36px`, `#111315`): Diagonal expand/fullscreen arrows icon (white)
  - **Bottom-Right Legend (Vertical Stack, `gap: 6px`):**
    - Blue dot: `>60%`
    - Green dot: `40–60%`
    - Amber dot: `20–40%`
    - Gray dot: `<20%`

### Card 3 (Bottom-Left): `Card_Possession_Momentum`
*Replaces "Details" streamgraph card in reference.*
- **Card Header:**
  - Title: `"Possession Momentum"` (`15px` SemiBold)
  - Right Action: Circular More Options button (`36px`, `#F6F7F9`).
- **Top Metric Strip (Horizontal Auto-Layout, `gap: 36px`):**
  - Metric 1: `"1,420"` (`20px` Bold) + `"Pass Sequences"` (`12px` `#6E7684`)
  - Metric 2: `"84%"` (`20px` Bold) + `"Retention Rate"` (`12px` `#6E7684`)
  - Metric 3: `"19"` (`20px` Bold) + `"Ball Recoveries"` (`12px` `#6E7684`)
- **Bottom Visualization (Symmetrical Streamgraph / Wave Chart):**
  - Horizontal multi-layered smooth wave (`#9CC3FC` outer wave, `#5B9BF8` mid wave, `#1D6FF2` core wave) expanding and contracting across 3 match intervals.
  - **3 Vertical Dashed Marker Lines** intersecting the wave:
    - **Marker 1 (Left — `15'`):** Top white pill `"54%"` / Bottom white pill `"46%"`
    - **Marker 2 (Center — `45'`):** Top **Solid Black Active Pill** `"74%"` (`#111315` fill, white text) / Bottom white pill `"26%"`
    - **Marker 3 (Right — `67'`):** Top white pill `"64%"` / Bottom white pill `"36%"`

### Card 4 (Bottom-Right): `Card_Probability_Trend`
*Replaces "New Request Trend" spline chart card in reference.*
- **Card Header:**
  - Title: `"Live Possession Probability"` (`15px` SemiBold)
  - Right Action: Circular More Options button (`36px`, `#F6F7F9`).
- **Card Body (Multi-Line Spline Chart):**
  - **Y-Axis:** Labels `0`, `20`, `40`, `60`, `80`, `100` (`11px` `#9CA3AF`) with faint horizontal gridlines (`#F1F3F6`).
  - **3 Smooth Cubic Spline Curves (`stroke-width: 2.5px`):**
    1. **Top Curve (`#1D6FF2` Blue):** Starts around `35`, rises smoothly to `86`, terminating in a solid Blue pill badge (`#1D6FF2`, white text): `"Home Team"`.
    2. **Middle Curve (`#38C976` Green):** Starts around `20`, rises to `62`, terminating in a solid Green pill badge (`#38C976`, white text): `"Away Team"`.
    3. **Bottom Curve (`#F5B749` Amber):** Starts around `16`, stays steady around `32`, terminating in a solid Amber pill badge (`#F5B749`, white text): `"Contested"`.
  - **Active Inspection Crosshair (at ~35% X-axis):**
    - Vertical dashed line (`#6E7684`).
    - Circular node on the Blue curve (`10px`, `#FFFFFF` fill, `3px` `#1D6FF2` stroke).
    - Floating **Solid Black Tooltip Pill** directly above the node (`#111315` fill, `9999px` radius): `"76"` in white + `"+4.2%"` in `#38C976` (emerald green).

---

## 5. Figcraft MCP / CLI Execution Steps for the Agent

Follow this exact sequence when executing Figcraft commands:
1. **Initialize Frame & Tokens:** Create the `1440x1024` root canvas frame (`#F6F7F9`) and register color/typography styles.
2. **Build Sidebar & Header:** Construct the left `64px` icon rail with exact active/inactive circular states and the top header with inline KPI typography.
3. **Construct Bento Grid:** Create the 2×2 Auto-Layout container (`gap: 20px`) with four `24px`-rounded white cards.
4. **Render Custom Vector Charts:**
   - Card 1: Draw the 3-segment rounded-cap 240° arc + dotted-leader legend rows.
   - Card 2: Draw the stylized 6-zone vector football pitch + floating center metric card + zoom/expand controls.
   - Card 3: Draw the 3-layer blue symmetrical streamgraph + 3 vertical dashed marker lines with top/bottom percentage pills.
   - Card 4: Draw the 3 smooth spline curves + right-side pill tags + active black floating tooltip badge (`76 +4.2%`).
5. **Visual Audit:** Verify padding consistency (`24px` internal card padding), text legibility, and visual alignment against the reference layout before finishing.