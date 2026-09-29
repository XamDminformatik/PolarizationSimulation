"""
Polarimetry Simulation: 3D Schematic Optics & Real-Time Stokes Parameters
Visualizes a pulsed laser beam passing through a rotating Half-Wave Plate (lambda/2)
and a fixed Quarter-Wave Plate (lambda/4), generating arbitrary polarization states.
Alongside the 3D optical rail, synchronized 2D plots display Stokes parameters S1 and S3.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Color palette: MBI Dark Navy Presentation Theme
C_BG = "#1B2436"         # Background canvas
C_PANEL = "#222C42"      # Card / panel background
C_TEXT = "#F7F7F7"       # Soft cream / white text
C_MUTED = "#8E9EB2"      # Muted text & tick labels
C_CYAN = "#3EC6E0"       # Primary cyan accent (linear wave / S1)
C_BLUE = "#5B7CFF"       # Vibrant blue
C_MAGENTA = "#FF4D6D"    # Magenta / coral accent (circular wave / S3)
C_GOLD = "#FFC857"       # Fast axis / tracking dots
C_MOUNT = "#4A5873"      # Optics ring / mount body
C_POST = "#607290"       # Optical post
C_AXIS = "#3A4760"       # Optical rail / beamline axis

def create_figure():
    fig = plt.figure(figsize=(16, 9), facecolor=C_BG)
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.60, 1.05],
        height_ratios=[1.0, 1.0],
        left=0.03, right=0.96, top=0.94, bottom=0.09,
        wspace=0.22, hspace=0.28
    )

    ax_3d = fig.add_subplot(gs[:, 0], projection='3d', facecolor=C_BG)
    ax_s1 = fig.add_subplot(gs[0, 1], facecolor=C_PANEL)
    ax_s3 = fig.add_subplot(gs[1, 1], facecolor=C_PANEL)

    return fig, ax_3d, ax_s1, ax_s3

def setup_3d_axes(ax_3d):
    ax_3d.set_box_aspect((3.8, 1.2, 1.2))
    ax_3d.view_init(elev=18, azim=-62)

    # Hide default background panes & grids
    ax_3d.xaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.yaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.zaxis.set_pane_color((0, 0, 0, 0))
    ax_3d.grid(False)
    ax_3d.set_axis_off()

    ax_3d.set_xlim(-0.2, 7.2)
    ax_3d.set_ylim(-1.6, 1.6)
    ax_3d.set_zlim(-1.7, 1.7)

    # Optical rail along z-axis (center line)
    z_rail = np.linspace(-0.2, 7.0, 100)
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.zeros_like(z_rail),
               color=C_AXIS, linestyle='--', linewidth=1.5, zorder=1)

    # Optical rail base track at y=-1.5
    ax_3d.plot(z_rail, np.zeros_like(z_rail), np.full_like(z_rail, -1.5),
               color=C_MOUNT, linestyle='-', linewidth=3.0, alpha=0.5, zorder=1)

    # Clean label at output
    ax_3d.text(6.6, 0.0, 1.45, "Pump Polarization",
               color=C_MAGENTA, fontsize=11.5, fontweight='bold', ha='center')

def draw_optical_mount(ax_3d, z_pos, label, color_glass, fast_axis_angle):
    """
    Renders a schematic optical mount:
    - Optical post clamped to table
    - Circular mount ring
    - Semi-transparent glass disc
    - Fast axis line
    - Minimal title (just lambda/2 or lambda/4)
    """
    R_outer = 1.05
    R_inner = 0.90
    theta = np.linspace(0, 2 * np.pi, 60)

    # 1. Optical Post
    post_z = [z_pos, z_pos]
    post_x = [0, 0]
    post_y = [-R_outer, -1.5]
    ax_3d.plot(post_z, post_x, post_y, color=C_POST, linewidth=4.5, solid_capstyle='round')

    # Post base clamp
    clamp_z = [z_pos - 0.15, z_pos + 0.15]
    clamp_x = [0, 0]
    clamp_y = [-1.5, -1.5]
    ax_3d.plot(clamp_z, clamp_x, clamp_y, color=C_MOUNT, linewidth=6.0, solid_capstyle='round')

    # 2. Outer Mount Ring
    ring_x = R_outer * np.cos(theta)
    ring_y = R_outer * np.sin(theta)
    ring_z = np.full_like(ring_x, z_pos)
    ax_3d.plot(ring_z, ring_x, ring_y, color=C_MOUNT, linewidth=3.0)

    # Inner aperture ring
    in_x = R_inner * np.cos(theta)
    in_y = R_inner * np.sin(theta)
    ax_3d.plot(ring_z, in_x, in_y, color=C_TEXT, linewidth=0.8, alpha=0.6)

    # 3. Glass Disc Surface (Translucent)
    r = np.linspace(0, R_inner, 15)
    TH, RR = np.meshgrid(theta, r)
    disc_x = RR * np.cos(TH)
    disc_y = RR * np.sin(TH)
    disc_z = np.full_like(disc_x, z_pos)
    ax_3d.plot_surface(disc_z, disc_x, disc_y, alpha=0.18, color=color_glass, edgecolor='none')

    # 4. Fast Axis Indicator Line
    L = R_inner * 0.88
    fa_x = [-L * np.cos(fast_axis_angle), L * np.cos(fast_axis_angle)]
    fa_y = [-L * np.sin(fast_axis_angle), L * np.sin(fast_axis_angle)]
    fa_z = [z_pos, z_pos]
    line_fa, = ax_3d.plot(fa_z, fa_x, fa_y, color=C_GOLD, linewidth=3.2,
                          solid_capstyle='round', zorder=10)

    # 5. Clean, uncluttered title (e.g. lambda/2 or lambda/4)
    ax_3d.text(z_pos, 0, R_outer + 0.32, label,
               color=C_TEXT, fontsize=14, fontweight='bold', ha='center')

    return line_fa

def draw_projection_screen(ax_3d, z_pos):
    """Semi-transparent detection/projection screen showing 2D transverse polarization ellipse."""
    S = 1.15
    y_screen = [-S, S, S, -S]
    z_screen = [S, S, -S, -S]
    x_screen = [z_pos, z_pos, z_pos, z_pos]
    verts = [list(zip(x_screen, y_screen, z_screen))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor=C_PANEL, edgecolor=C_MUTED, linewidth=1.2)
    ax_3d.add_collection3d(poly)

    # Crosshairs on the screen
    ax_3d.plot([z_pos, z_pos], [-S, S], [0, 0], color=C_MUTED, linestyle=':', linewidth=1.0, alpha=0.7)
    ax_3d.plot([z_pos, z_pos], [0, 0], [-S, S], color=C_MUTED, linestyle=':', linewidth=1.0, alpha=0.7)

    # Dynamic polarization ellipse on the screen
    line_ellipse, = ax_3d.plot([], [], [], color=C_MAGENTA, linewidth=2.5, alpha=0.9)
    dot_tip, = ax_3d.plot([], [], [], marker='o', markersize=6, color=C_GOLD, markeredgecolor='white')

    return line_ellipse, dot_tip

def setup_2d_plots(ax_s1, ax_s3):
    phi_dense = np.linspace(0, 90, 400)
    phi_rad = np.deg2rad(phi_dense)

    # Theoretical curves
    S1_theory = np.cos(4 * phi_rad)
    S3_theory = np.sin(4 * phi_rad)

    for ax in (ax_s1, ax_s3):
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values():
            spine.set_color(C_MUTED)
            spine.set_alpha(0.4)
        ax.tick_params(colors=C_MUTED, labelsize=9.5)
        ax.set_xlim(0, 90)
        ax.set_ylim(-1.25, 1.25)
        ax.axhline(0, color=C_MUTED, linestyle=':', linewidth=0.9, alpha=0.6)
        ax.set_xticks([0, 22.5, 45, 67.5, 90])
        ax.grid(True, linestyle=':', alpha=0.25, color=C_MUTED)

    # Top Plot: Stokes S1 (Linear Polarization Parameter)
    ax_s1.set_xticklabels([])
    ax_s1.set_ylabel(r"$S_1$: Linear Polarization Angle", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_s1.set_yticks([-1.0, 0.0, 1.0])
    ax_s1.set_yticklabels(["Linear Vertical", "0", "Linear Horizontal"])
    ax_s1.axhline(1.0, color=C_CYAN, linestyle='--', linewidth=0.8, alpha=0.4)
    ax_s1.axhline(-1.0, color=C_CYAN, linestyle='--', linewidth=0.8, alpha=0.4)
    ax_s1.plot(phi_dense, S1_theory, color=C_CYAN, linewidth=2.2, alpha=0.9)
    marker_s1, = ax_s1.plot([], [], 'o', color=C_GOLD, markersize=8.5, markeredgecolor='white', markeredgewidth=1.5)

    # Bottom Plot: Stokes S3 (Circular Polarization / Helicity Parameter)
    ax_s3.set_xticklabels([
        "0°\nLinear Horizontal",
        "22.5°\nCircular (σ⁺)",
        "45°\nLinear Vertical",
        "67.5°\nCircular (σ⁻)",
        "90°\nLinear Horizontal"
    ])
    ax_s3.set_xlabel(r"$\lambda/2$ Plate Rotation Angle $\phi$", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_s3.set_ylabel(r"$S_3$: Ellipticity", color=C_TEXT, fontsize=11, fontweight='bold')
    ax_s3.set_yticks([-1.0, 0.0, 1.0])
    ax_s3.set_yticklabels(["Circular (σ⁻)", "Linear", "Circular (σ⁺)"])
    ax_s3.axhline(1.0, color=C_MAGENTA, linestyle='--', linewidth=0.8, alpha=0.4)
    ax_s3.axhline(-1.0, color=C_MAGENTA, linestyle='--', linewidth=0.8, alpha=0.4)
    ax_s3.plot(phi_dense, S3_theory, color=C_MAGENTA, linewidth=2.2, alpha=0.9)
    marker_s3, = ax_s3.plot([], [], 'o', color=C_GOLD, markersize=8.5, markeredgecolor='white', markeredgewidth=1.5)

    return marker_s1, marker_s3

def build_simulation(output_path=None, preview_only=False, fps=30, duration_sec=10.0, pulse_speed=1.10):
    fig, ax_3d, ax_s1, ax_s3 = create_figure()
    setup_3d_axes(ax_3d)

    # Geometry constants
    z_hwp = 2.0
    z_qwp = 4.2
    z_screen = 6.6
    E_amp = 0.82

    # Pulse train parameters (harmonized for seamless boundary condition when T=10s, v=1.10)
    D = 2.2                 # Distance between consecutive pulses in train
    sigma = 0.32            # Gaussian pulse width (approx 2-3 optical cycles)
    lambda_0 = 0.55         # Optical carrier wavelength
    k = 2 * np.pi / lambda_0
    v = pulse_speed         # Pulse travel speed along z (calm, easy-to-follow propagation)
    omega = k * v           # Group velocity = phase velocity (non-dispersive propagation)

    # Draw Mounts with minimal titles
    line_fa_hwp = draw_optical_mount(
        ax_3d, z_hwp, r"$\lambda/2$", C_CYAN, 0.0
    )
    draw_optical_mount(
        ax_3d, z_qwp, r"$\lambda/4$", C_BLUE, 0.0
    )

    # Draw Projection Screen at output
    line_screen_ellipse, dot_screen_tip = draw_projection_screen(ax_3d, z_screen)

    # Wave line objects
    line_wave_0, = ax_3d.plot([], [], [], color=C_CYAN, linewidth=2.4)
    line_wave_1, = ax_3d.plot([], [], [], color=C_BLUE, linewidth=2.4)
    line_wave_2, = ax_3d.plot([], [], [], color=C_MAGENTA, linewidth=2.6)

    # 2D plot tracking markers
    marker_s1, marker_s3 = setup_2d_plots(ax_s1, ax_s3)

    # High-density spatial discretization points for smooth wave rendering
    pts_0 = np.linspace(0.0, z_hwp, 140)
    pts_1 = np.linspace(z_hwp, z_qwp, 150)
    pts_2 = np.linspace(z_qwp, z_screen, 160)
    tau_ellipse = np.linspace(0, 2 * np.pi, 80)

    total_frames = int(fps * duration_sec)

    def calc_envelope(z_arr, t_val):
        """Calculates smooth Gaussian pulse train envelope."""
        pos = (v * t_val) % D
        env = np.zeros_like(z_arr)
        for n in range(-2, 5):
            c = pos + n * D
            env += np.exp(-0.5 * ((z_arr - c) / sigma)**2)
        return env

    def get_state(frame_idx):
        progress = (frame_idx % total_frames) / float(total_frames)
        phi_deg = progress * 90.0
        phi_rad = np.deg2rad(phi_deg)
        t = progress * duration_sec

        s1 = np.cos(4 * phi_rad)
        s3 = np.sin(4 * phi_rad)
        return phi_deg, phi_rad, t, s1, s3

    def update(frame_idx):
        phi_deg, phi_rad, t, s1, s3 = get_state(frame_idx)

        # 1. Update HWP Fast Axis Line in 3D
        R_fa = 0.90 * 0.88
        fa_x = [-R_fa * np.cos(phi_rad), R_fa * np.cos(phi_rad)]
        fa_y = [-R_fa * np.sin(phi_rad), R_fa * np.sin(phi_rad)]
        line_fa_hwp.set_data(fa_x, fa_y)
        line_fa_hwp.set_3d_properties([z_hwp, z_hwp], zdir='x')

        # 2. Update Pulsed Wave Propagation
        # Segment 0: Before HWP (Horizontally polarized linear pulse packet)
        env_0 = calc_envelope(pts_0, t)
        phase_0 = k * pts_0 - omega * t
        Ex0 = env_0 * E_amp * np.cos(phase_0)
        Ey0 = np.zeros_like(pts_0)
        line_wave_0.set_data(Ex0, Ey0)
        line_wave_0.set_3d_properties(pts_0, zdir='x')

        # Segment 1: Between HWP and QWP (Linear pulse packet rotated by 2*phi)
        env_1 = calc_envelope(pts_1, t)
        phase_1 = k * pts_1 - omega * t
        Ex1 = env_1 * E_amp * np.cos(2 * phi_rad) * np.cos(phase_1)
        Ey1 = env_1 * E_amp * np.sin(2 * phi_rad) * np.cos(phase_1)
        line_wave_1.set_data(Ex1, Ey1)
        line_wave_1.set_3d_properties(pts_1, zdir='x')

        # Segment 2: After QWP (Quarter-wave retardation: corkscrew pulse packet or linear)
        env_2 = calc_envelope(pts_2, t)
        phase_2 = k * pts_2 - omega * t
        Ex2 = env_2 * E_amp * np.cos(2 * phi_rad) * np.cos(phase_2)
        Ey2 = env_2 * E_amp * np.sin(2 * phi_rad) * np.sin(phase_2)
        line_wave_2.set_data(Ex2, Ey2)
        line_wave_2.set_3d_properties(pts_2, zdir='x')

        # 3. Update Output Screen Polarization Ellipse
        ellipse_x = E_amp * np.cos(2 * phi_rad) * np.cos(tau_ellipse)
        ellipse_y = E_amp * np.sin(2 * phi_rad) * np.sin(tau_ellipse)
        line_screen_ellipse.set_data(ellipse_x, ellipse_y)
        line_screen_ellipse.set_3d_properties(np.full_like(tau_ellipse, z_screen), zdir='x')

        # Instantaneous tip on screen
        env_screen = calc_envelope(np.array([z_screen]), t)[0]
        phase_screen = k * z_screen - omega * t
        tip_x = env_screen * E_amp * np.cos(2 * phi_rad) * np.cos(phase_screen)
        tip_y = env_screen * E_amp * np.sin(2 * phi_rad) * np.sin(phase_screen)
        dot_screen_tip.set_data([tip_x], [tip_y])
        dot_screen_tip.set_3d_properties([z_screen], zdir='x')

        # 4. Update 2D Tracking Points
        marker_s1.set_data([phi_deg], [s1])
        marker_s3.set_data([phi_deg], [s3])

        return (
            line_fa_hwp,
            line_wave_0, line_wave_1, line_wave_2,
            line_screen_ellipse, dot_screen_tip,
            marker_s1, marker_s3
        )

    if preview_only:
        update(int(total_frames * 0.25))
        preview_file = Path("polarimetry_preview.png")
        fig.savefig(preview_file, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"Generated preview image: {preview_file.resolve()}")
        return

    if output_path is not None:
        out_p = Path(output_path)
        print(f"Exporting animation to {out_p.name} ({total_frames} frames @ {fps} fps)...")
        ani = FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / fps, blit=False)

        if out_p.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps, metadata=dict(artist='Max Dittmer'), bitrate=4500)
            ani.save(out_p, writer=writer, dpi=120)
        elif out_p.suffix.lower() == ".gif":
            writer = PillowWriter(fps=fps)
            ani.save(out_p, writer=writer)
        else:
            raise ValueError(f"Unsupported extension: {out_p.suffix}")

        plt.close(fig)
        print(f"Successfully saved animation to {out_p.resolve()}")
        return

    ani = FuncAnimation(fig, update, frames=total_frames, interval=1000.0 / fps, blit=False)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polarimetry 3D Simulation & Stokes Parameter Evolution")
    parser.add_argument("--preview", action="store_true", help="Save a single high-res preview frame")
    parser.add_argument("--output", type=str, default=None, help="Output filename (.mp4 or .gif)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of full cycle in seconds")
    parser.add_argument("--speed", type=float, default=1.10, help="Pulse propagation speed along optical axis")
    args = parser.parse_args()

    build_simulation(
        output_path=args.output,
        preview_only=args.preview,
        fps=args.fps,
        duration_sec=args.duration,
        pulse_speed=args.speed
    )
