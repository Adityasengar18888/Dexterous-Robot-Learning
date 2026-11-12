import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_plots():
    try:
        plots_dir = "results/phase13_5/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Read sample CSVs
        s_csv = "results/phase13_5/d5_lift_trajectories/success_sample_1_seed42_ep5001.csv"
        f_csv = "results/phase13_5/d5_lift_trajectories/grasp_but_no_lift_sample_1_seed42_ep5002.csv"
        
        df_s = pd.read_csv(s_csv)
        df_f = pd.read_csv(f_csv)
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
        
        # 1. Altitude (Z)
        axes[0, 0].plot(df_s["step"], df_s["ee_pos_2"], "b-", linewidth=2, label="End-Effector Z")
        axes[0, 0].plot(df_s["step"], df_s["cube_z"], "g-", linewidth=2, label="Cube Z (Lifted)")
        axes[0, 0].axhline(y=0.05, color="r", linestyle="--", label="Lift Threshold (0.05m)")
        axes[0, 0].set_title("Successful Episode: Altitude Profile", fontsize=12, fontweight="bold")
        axes[0, 0].set_ylabel("Altitude (m)")
        axes[0, 0].legend(loc="upper left")
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df_f["step"], df_f["ee_pos_2"], "b-", linewidth=2, label="End-Effector Z")
        axes[0, 1].plot(df_f["step"], df_f["cube_z"], "r-", linewidth=2, label="Cube Z (Slip & Drop)")
        axes[0, 1].axhline(y=0.05, color="r", linestyle="--", label="Lift Threshold (0.05m)")
        axes[0, 1].set_title("Failed Episode (GRASP_BUT_NO_LIFT): Altitude Profile", fontsize=12, fontweight="bold")
        axes[0, 1].set_ylabel("Altitude (m)")
        axes[0, 1].legend(loc="upper left")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2. Contact Normal Forces
        axes[1, 0].plot(df_s["step"], df_s["f9_normal_force"], "c-", linewidth=1.5, label="Finger 9 Force")
        axes[1, 0].plot(df_s["step"], df_s["f10_normal_force"], "m-", linewidth=1.5, label="Finger 10 Force")
        axes[1, 0].plot(df_s["step"], df_s["total_normal_force"], "k--", linewidth=1.5, label="Total Normal Force")
        axes[1, 0].set_title("Successful Episode: Contact Normal Force", fontsize=12, fontweight="bold")
        axes[1, 0].set_ylabel("Force (N)")
        axes[1, 0].legend(loc="upper left")
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_f["step"], df_f["f9_normal_force"], "c-", linewidth=1.5, label="Finger 9 Force")
        axes[1, 1].plot(df_f["step"], df_f["f10_normal_force"], "m-", linewidth=1.5, label="Finger 10 Force")
        axes[1, 1].plot(df_f["step"], df_f["total_normal_force"], "k--", linewidth=1.5, label="Total Normal Force")
        axes[1, 1].set_title("Failed Episode: Contact Normal Force (Slip at Step 22)", fontsize=12, fontweight="bold")
        axes[1, 1].set_ylabel("Force (N)")
        axes[1, 1].legend(loc="upper left")
        axes[1, 1].grid(True, alpha=0.3)
        
        # 3. Relative EE-Cube Distance
        axes[2, 0].plot(df_s["step"], df_s["rel_dist"], "g-", linewidth=2, label="EE-Cube Distance")
        axes[2, 0].axhline(y=0.05, color="gray", linestyle=":", label="Grasp Sphere (0.05m)")
        axes[2, 0].set_title("Successful Episode: Relative Distance (Stable Transport)", fontsize=12, fontweight="bold")
        axes[2, 0].set_xlabel("Episode Step")
        axes[2, 0].set_ylabel("Distance (m)")
        axes[2, 0].legend(loc="upper left")
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(df_f["step"], df_f["rel_dist"], "r-", linewidth=2, label="EE-Cube Distance")
        axes[2, 1].axhline(y=0.05, color="gray", linestyle=":", label="Grasp Sphere (0.05m)")
        axes[2, 1].set_title("Failed Episode: Relative Distance (Divergence at Step 22)", fontsize=12, fontweight="bold")
        axes[2, 1].set_xlabel("Episode Step")
        axes[2, 1].set_ylabel("Distance (m)")
        axes[2, 1].legend(loc="upper left")
        axes[2, 1].grid(True, alpha=0.3)
        
        plot_path = f"{plots_dir}/lift_trajectory_comparison.png"
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Plot successfully saved to: {plot_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_plots()

