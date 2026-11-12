from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

def draw_trajectory_plot():
    # Load data
    s_csv = "results/phase13_5/d5_lift_trajectories/success_sample_1_seed42_ep5001.csv"
    f_csv = "results/phase13_5/d5_lift_trajectories/grasp_but_no_lift_sample_1_seed42_ep5002.csv"
    df_s = pd.read_csv(s_csv)
    df_f = pd.read_csv(f_csv)
    
    # Create high-res canvas
    w, h = 1200, 1000
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Title
    draw.text((30, 20), "Phase 13.5: Lift Trajectory Telemetry Comparison", fill=(0, 0, 0))
    draw.text((30, 45), "Left: Successful Lift Episode | Right: GRASP_BUT_NO_LIFT Failure (Slip at Step 22)", fill=(100, 100, 100))
    
    # Panel layout: 3 rows, 2 columns
    # Row 1: Altitude (EE Z & Cube Z)
    # Row 2: Normal Forces (F9 & F10)
    # Row 3: Relative Distance (EE to Cube)
    
    def draw_panel(x0, y0, pw, ph, steps_s, y_series_list, labels, colors, title, y_range, thresh=None):
        # Draw background and axes
        draw.rectangle([x0, y0, x0+pw, y0+ph], outline=(200, 200, 200), fill=(250, 250, 250))
        draw.text((x0+10, y0+10), title, fill=(0, 0, 0))
        
        # Grid lines
        for gi in range(5):
            gy = y0 + ph - int(gi * ph / 4)
            val = y_range[0] + gi * (y_range[1] - y_range[0]) / 4
            draw.line([x0, gy, x0+pw, gy], fill=(230, 230, 230))
            draw.text((x0+5, gy-12), f"{val:.2f}", fill=(120, 120, 120))
            
        # Threshold line
        if thresh is not None:
            ty = y0 + ph - int((thresh - y_range[0]) / (y_range[1] - y_range[0]) * ph)
            if y0 <= ty <= y0+ph:
                draw.line([x0, ty, x0+pw, ty], fill=(255, 0, 0), width=2)
                draw.text((x0+pw-140, ty-14), f"Threshold ({thresh})", fill=(255, 0, 0))
                
        # Draw series
        max_step = 100
        for y_vals, col, lab in zip(y_series_list, colors, labels):
            points = []
            for s, yv in zip(steps_s, y_vals):
                px = x0 + int(s / max_step * pw)
                norm_y = (yv - y_range[0]) / (y_range[1] - y_range[0] + 1e-6)
                norm_y = max(0.0, min(1.0, norm_y))
                py = y0 + ph - int(norm_y * ph)
                points.append((px, py))
            for pi in range(len(points)-1):
                draw.line([points[pi], points[pi+1]], fill=col, width=2)
                
        # Legend
        lx = x0 + pw - 180
        ly = y0 + 10
        for lab, col in zip(labels, colors):
            draw.rectangle([lx, ly, lx+15, ly+10], fill=col)
            draw.text((lx+20, ly-2), lab, fill=(0, 0, 0))
            ly += 16
            
    # Row 1: Altitude
    draw_panel(50, 80, 520, 260, df_s["step"], [df_s["ee_pos_2"], df_s["cube_z"]], ["EE Z", "Cube Z"], [(0, 0, 200), (0, 180, 0)], "Success: Altitude Profile (m)", [0.0, 0.25], thresh=0.05)
    draw_panel(630, 80, 520, 260, df_f["step"], [df_f["ee_pos_2"], df_f["cube_z"]], ["EE Z", "Cube Z (Slip)"], [(0, 0, 200), (220, 0, 0)], "Failure: Altitude Profile (m)", [0.0, 0.25], thresh=0.05)
    
    # Row 2: Contact Force
    draw_panel(50, 380, 520, 260, df_s["step"], [df_s["f9_normal_force"], df_s["f10_normal_force"]], ["Finger 9", "Finger 10"], [(0, 150, 200), (200, 0, 150)], "Success: Normal Forces (N)", [0.0, 80.0])
    draw_panel(630, 380, 520, 260, df_f["step"], [df_f["f9_normal_force"], df_f["f10_normal_force"]], ["Finger 9", "Finger 10"], [(0, 150, 200), (200, 0, 150)], "Failure: Normal Forces (Slip @ Step 22)", [0.0, 80.0])
    
    # Row 3: Relative Distance
    draw_panel(50, 680, 520, 260, df_s["step"], [df_s["rel_dist"]], ["EE-Cube Dist"], [(0, 150, 0)], "Success: Relative Distance (m)", [0.0, 0.3], thresh=0.05)
    draw_panel(630, 680, 520, 260, df_f["step"], [df_f["rel_dist"]], ["EE-Cube Dist"], [(220, 0, 0)], "Failure: Relative Distance (Divergence)", [0.0, 0.3], thresh=0.05)
    
    img.save("results/phase13_5/plots/lift_trajectory_comparison.png")
    print("Pillow plot successfully saved to results/phase13_5/plots/lift_trajectory_comparison.png", flush=True)

if __name__ == "__main__":
    draw_trajectory_plot()

