# Project Directory Structure

```text
.
|   .gitignore
|   codebase.md
|   README.md
|   readme2.md
|   requirements.txt
|   
+---data
|   +---outputs
|   |       log.csv
|   |       
|   \---vedios
|           easy.mp4
|           h1.mp4
|           h2.mp4
|           jagganath.mp4
|           medium.mp4
|           
+---models
|       head_yolov8.pt
|       
\---src
        collect.py
        config.py
        features.py
        __init__.py
```


### `metric.csv` Full Data Dictionary

The `metric.csv` file generates a comprehensive dataset containing the raw tracking logs alongside derived statistical features over 5-second and 10-second rolling windows. 

| Exact Column Name(s) | Category | Description |
| :--- | :--- | :--- |
| **`t`** | **Time** | Video timestamp in seconds. |
| **`count`** | **Raw Density** | Raw number of detected head boxes from YOLO. |
| `count_sma5`, `count_sma10` | Density Trend | 5-sec and 10-sec moving average of crowd size. |
| `count_res5`, `count_res10` | Density Spike | Sudden increase/decrease in people vs. recent baseline. |
| `count_std5`, `count_std10` | Density Volatility | Standard deviation; indicates rapidly changing crowd size. |
| `count_z5`, `count_z10` | Density Anomaly | Z-score; mathematical severity of a crowd count spike. |
| **`cov`** | **Raw Coverage** | Raw frame area coverage ratio (0.0 to 1.0). |
| `cov_sma5`, `cov_sma10` | Coverage Trend | Moving average of occupied physical space in the frame. |
| `cov_res5`, `cov_res10` | Coverage Spike | Sudden changes in occupied physical space. |
| `cov_std5`, `cov_std10` | Coverage Volatility | Fluctuation in physical space occupation. |
| `cov_z5`, `cov_z10` | Coverage Anomaly | Z-score; mathematical severity of a coverage spike. |
| **`wcount`** | **Raw Weighted Count** | Raw perspective-weighted count; lower/closer heads contribute more. |
| `wcount_sma5`, `wcount_sma10` | Weighted Trend | Moving average of perspective-weighted crowd mass. |
| `wcount_res5`, `wcount_res10` | Weighted Spike | Sudden influx of people specifically in the foreground/lower frame. |
| `wcount_std5`, `wcount_std10` | Weighted Volatility | Fluctuation of crowd mass in the perspective foreground. |
| `wcount_z5`, `wcount_z10` | Weighted Anomaly | Z-score; mathematical severity of a foreground mass spike. |
| **`speed`** | **Raw Speed** | Raw mean KLT optical flow speed in pixels/second. |
| `speed_sma5`, `speed_sma10` | Speed Trend | Moving average of overall crowd velocity. |
| `speed_res5`, `speed_res10` | Speed Spike | Sudden burst of movement (e.g., running, panic, or fast vehicles). |
| `speed_std5`, `speed_std10` | Speed Volatility | Fluctuation in movement speed. |
| `speed_z5`, `speed_z10` | Speed Anomaly | Z-score; mathematical severity of a sudden speed change. |
| **`dirx`** | **Raw X-Direction** | Raw horizontal motion vector (positive = right, negative = left). |
| `dirx_sma5`, `dirx_sma10` | X-Direction Trend | Moving average of horizontal flow. |
| `dirx_res5`, `dirx_res10` | X-Direction Spike | Sudden shift in horizontal movement. |
| `dirx_std5`, `dirx_std10` | X-Direction Volatility | Instability in horizontal crowd flow. |
| `dirx_z5`, `dirx_z10` | X-Direction Anomaly | Z-score; mathematical severity of horizontal directional shift. |
| **`diry`** | **Raw Y-Direction** | Raw vertical motion vector (positive = down, negative = up). |
| `diry_sma5`, `diry_sma10` | Y-Direction Trend | Moving average of vertical flow. |
| `diry_res5`, `diry_res10` | Y-Direction Spike | Sudden shift in vertical movement. |
| `diry_std5`, `diry_std10` | Y-Direction Volatility | Instability in vertical crowd flow. |
| `diry_z5`, `diry_z10` | Y-Direction Anomaly | Z-score; mathematical severity of vertical directional shift. |
| **`consist`** | **Raw Consistency** | Raw directional consistency (0.0 = chaotic scattering, 1.0 = uniform flow). |
| `consist_sma5`, `consist_sma10` | Consistency Trend | Moving average of crowd organization. |
| `consist_res5`, `consist_res10` | Consistency Spike | Sudden shift from organized movement to chaotic scattering (or vice versa). |
| `consist_std5`, `consist_std10` | Consistency Volatility | Fluctuation in crowd organization/chaos. |
| `consist_z5`, `consist_z10` | Consistency Anomaly | Z-score; mathematical severity of a breakdown in crowd flow. |
| **`g0` ... `g15`** | **Raw Grid Matrix** | 16 individual columns (`g0`, `g1`, `g2` ... `g15`) representing raw head counts inside a 4x4 spatial grid. |

*(Note: Grid columns `g0`-`g15` remain raw unless the script is explicitly run with the `--smooth-grid` flag, which will additionally generate `_sma`, `_res`, `_std`, and `_z` features for all 16 spatial zones).*