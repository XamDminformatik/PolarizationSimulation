import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

def retarder_matrix(gamma, theta):
    """
    Jones matrix for a linear retarder with retardation gamma
    and fast axis at angle theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    R_inv = np.array([[c, s], [-s, c]])

    # Phase matrix
    P = np.array([[np.exp(1j * gamma / 2), 0],
                  [0, np.exp(-1j * gamma / 2)]])

    return R @ P @ R_inv

def hwp_matrix(theta):
    """Jones matrix for a Half-Wave Plate."""
    return retarder_matrix(np.pi, theta)

def qwp_matrix(theta):
    """Jones matrix for a Quarter-Wave Plate."""
    return retarder_matrix(np.pi / 2, theta)


def simulate_propagation(hwp_angle, qwp_angle):
    """
    Calculate the Jones vector of the probe pulse at different stages.
    """
    # 1. Initial state (s-polarized, usually y-axis)
    E0 = np.array([0, 1], dtype=complex)

    # 2. After HWP
    J_hwp = hwp_matrix(hwp_angle)
    E1 = J_hwp @ E0

    # 3. After QWP
    J_qwp = qwp_matrix(qwp_angle)
    E2 = J_qwp @ E1

    return E0, E1, E2

def generate_wave(E, z_start, z_end, t, num_points=100, k=2*np.pi, omega=2*np.pi):
    """Generate 3D wave points for a given Jones vector."""
    z = np.linspace(z_start, z_end, num_points)
    # E = [Ex, Ey]
    # E(z, t) = Re{ E * exp(i(kz - omega*t)) }
    phase = k * z - omega * t
    Ex = np.real(E[0] * np.exp(1j * phase))
    Ey = np.real(E[1] * np.exp(1j * phase))
    return z, Ex, Ey

def main():
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
    ax = fig.add_subplot(gs[:, 0], projection='3d', proj_type='ortho')

    ax_s1 = fig.add_subplot(gs[0, 1])
    ax_s3 = fig.add_subplot(gs[1, 1])

    plt.subplots_adjust(bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=0.3)

    # Setup parameters
    z_hwp = 2.0
    z_qwp = 4.0

    z_end = 6.0

    # Initial state
    hwp_angle_init = np.pi / 8  # 22.5 deg
    qwp_angle_init = 0.0


    # Lines for wave segments
    line_0, = ax.plot([], [], [], color='blue', label='Initial (s-pol)')
    line_1, = ax.plot([], [], [], color='green', label='After HWP')
    line_2, = ax.plot([], [], [], color='red', label='After QWP')

    # 2D Plot setups
    hwp_angles_plot = np.linspace(0, np.pi, 100)
    line_s1, = ax_s1.plot([], [], color='blue')
    point_s1, = ax_s1.plot([], [], 'ro')
    ax_s1.set_xlim(0, np.pi)
    ax_s1.set_ylim(-1.1, 1.1)
    ax_s1.set_xlabel('HWP Angle (rad)')
    ax_s1.set_ylabel('S1 (Linear)')
    ax_s1.grid(True)

    line_s3, = ax_s3.plot([], [], color='red')
    point_s3, = ax_s3.plot([], [], 'bo')
    ax_s3.set_xlim(0, np.pi)
    ax_s3.set_ylim(-1.1, 1.1)
    ax_s3.set_xlabel('HWP Angle (rad)')
    ax_s3.set_ylabel('S3 (Ellipticity)')
    ax_s3.grid(True)


    ax.set_xlim(0, z_end)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('Propagation (z)')
    ax.set_ylabel('Ex')
    ax.set_zlabel('Ey')
    ax.legend()
    ax.set_axis_off()

    # Draw optical elements
    def draw_plate(z, name, color):
        theta = np.linspace(0, 2*np.pi, 50)
        r = np.linspace(0, 1.5, 2)
        T, R = np.meshgrid(theta, r)
        Y = R * np.cos(T)
        X = R * np.sin(T)
        Z = np.full_like(X, z)
        ax.plot_surface(Z, X, Y, alpha=0.3, color=color)
        ax.text(z, 0, 1.6, name, ha='center')

    draw_plate(z_hwp, 'HWP', 'cyan')
    draw_plate(z_qwp, 'QWP', 'magenta')


    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_hwp = plt.axes([0.15, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_qwp = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    s_hwp = Slider(ax_hwp, 'HWP Angle (rad)', 0.0, np.pi, valinit=hwp_angle_init)
    s_qwp = Slider(ax_qwp, 'QWP Angle (rad)', 0.0, np.pi, valinit=qwp_angle_init)

    def update(frame):
        t = frame / 20.0
        hwp_angle = s_hwp.val
        qwp_angle = s_qwp.val
        E0, E1, E2 = simulate_propagation(hwp_angle, qwp_angle)

        z0, Ex0, Ey0 = generate_wave(E0, 0, z_hwp, t)
        line_0.set_data(z0, Ex0)
        line_0.set_3d_properties(Ey0)

        z1, Ex1, Ey1 = generate_wave(E1, z_hwp, z_qwp, t)
        line_1.set_data(z1, Ex1)
        line_1.set_3d_properties(Ey1)

        z2, Ex2, Ey2 = generate_wave(E2, z_qwp, z_end, t)
        line_2.set_data(z2, Ex2)
        line_2.set_3d_properties(Ey2)

        # Update Stokes parameters plots
        S1_vals = []
        S3_vals = []
        for h in hwp_angles_plot:
            _, _, E_test = simulate_propagation(h, qwp_angle)
            s0 = np.abs(E_test[0])**2 + np.abs(E_test[1])**2
            s1 = np.abs(E_test[0])**2 - np.abs(E_test[1])**2
            s3 = 2 * np.imag(E_test[0] * np.conj(E_test[1]))
            if s0 != 0:
                S1_vals.append(s1/s0)
                S3_vals.append(s3/s0)
            else:
                S1_vals.append(0)
                S3_vals.append(0)

        line_s1.set_data(hwp_angles_plot, S1_vals)
        line_s3.set_data(hwp_angles_plot, S3_vals)

        s0_curr = np.abs(E2[0])**2 + np.abs(E2[1])**2
        s1_curr = np.abs(E2[0])**2 - np.abs(E2[1])**2
        s3_curr = 2 * np.imag(E2[0] * np.conj(E2[1]))

        if s0_curr != 0:
            point_s1.set_data([hwp_angle], [s1_curr/s0_curr])
            point_s3.set_data([hwp_angle], [s3_curr/s0_curr])
        else:
            point_s1.set_data([hwp_angle], [0])
            point_s3.set_data([hwp_angle], [0])

        return line_0, line_1, line_2, line_s1, point_s1, line_s3, point_s3

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.show()

if __name__ == "__main__":
    main()
