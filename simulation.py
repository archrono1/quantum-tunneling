import sys
import numpy as _np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb


# 0. RUNTIME CONFIG

print("="*60)
print("Barrier Configuration:")
print("  0. No barriers (free propagation)")
print("  1. Single barrier")
print("  2. Double barrier")
print("="*60)
num_barriers = int(input("Select number of barriers (0/1/2): ").strip())
while num_barriers not in [0, 1, 2]:
    num_barriers = int(input("Invalid choice. Select 0, 1, or 2: ").strip())

print("="*60)
print("Grid Resolution:")
print("  256  (medium)")
print("  512  (fine, slow)")
print("  1024 (high definition, slowest)")
print("="*60)
N = int(input("Select grid resolution (256/512/1024): ").strip())
while N not in [256, 512, 1024]:
    N = int(input("Invalid choice. Select 256, 512, or 1024: ").strip())

print("="*60)
print("Compute Backend:")
print("  numpy  (CPU)")
print("  cupy   (GPU — requires NVIDIA GPU + CUDA)")
print("="*60)
backend = input("Select backend (numpy/cupy): ").strip().lower()
while backend not in ['numpy', 'cupy']:
    backend = input("Invalid choice. Select numpy or cupy: ").strip().lower()

if backend == 'cupy':
    import cupy as xp
else:
    xp = _np


# 1. MATPLOTLIB CONFIG

plt.style.use('dark_background')
plt.rcParams['text.color'] = '#00ff00'
plt.rcParams['axes.labelcolor'] = '#00ff00'
plt.rcParams['xtick.color'] = '#00ff00'
plt.rcParams['ytick.color'] = '#00ff00'
plt.rcParams['axes.titlecolor'] = '#00ff00'
plt.rcParams['axes.facecolor'] = '#000000'
plt.rcParams['figure.facecolor'] = '#000000'
plt.rcParams['axes.edgecolor'] = '#333333'


# 2. PHYSICAL CONSTANTS (Atomic Units: hbar = m_e = e = 1)

hbar = 1.0
m_e  = 1.0

# Unit conversion factors (to atomic units)
Å  = 1.8897261246257702   # 1 Angstrom in Bohr radii (a₀)
eV = 0.03674932217565499  # 1 electron-volt in Hartree (Eₕ)
# 1 atomic unit of time = hbar/Eₕ ≈ 24.188 attoseconds


# 3. SIMULATION GRID & POTENTIAL SETUP

extent = 700 * Å
x = xp.linspace(-extent/2, extent/2, N, endpoint=False)
y = xp.linspace(-extent/2, extent/2, N, endpoint=False)
X, Y = xp.meshgrid(x, y)

# Barrier geometry
a  = 3 * Å       # barrier width
b  = 5 * Å       # inter-barrier gap
V0 = 1.7 * eV   # barrier height

V = xp.zeros((N, N), dtype=xp.complex64)

if num_barriers >= 1:
    V = xp.where((X > 0) & (X < a), V0.astype(xp.complex64) if hasattr(V0, 'astype') else xp.complex64(V0), V)

if num_barriers >= 2:
    V = xp.where((X > a + b) & (X < a + b + a), V0.astype(xp.complex64) if hasattr(V0, 'astype') else xp.complex64(V0), V)

# 4. INITIAL WAVEFUNCTION SETUP

E0   = 0.8 * eV
σ    = 25.0 * Å
k_x0 = xp.sqrt(2 * m_e * E0) / hbar

# FIX 3 (dtype): use complex64 to halve RAM; float32 real ops promote correctly
psi = (
    xp.exp(-1/(4 * σ**2) * ((X + 150*Å)**2 + Y**2)) / xp.sqrt(2 * xp.pi * σ**2)
    * xp.exp(1j * k_x0 * X)
).astype(xp.complex64)

# FIX 4: enforce ∫|ψ|² dxdy = 1 on the initial state
dx = x[1] - x[0]
dy = y[1] - y[0]

def normalize(psi_arr):
    norm = xp.sqrt(xp.sum(xp.abs(psi_arr)**2) * dx * dy)
    return psi_arr / norm

psi = normalize(psi)


# 5. SPLIT-STEP FOURIER 


TOTAL_TIME_AU = 4000   
num_steps      = 8000
dt             = TOTAL_TIME_AU / num_steps
store_steps    = 800
steps_per_frame = num_steps // store_steps

kx = 2 * xp.pi * xp.fft.fftfreq(N, d=float(dx))
ky = 2 * xp.pi * xp.fft.fftfreq(N, d=float(dy))
Kx, Ky = xp.meshgrid(kx, ky)

T = (hbar**2 / (2 * m_e)) * (Kx**2 + Ky**2)

exp_V = xp.exp(-1j * V * (dt / 2.0) / hbar).astype(xp.complex64)
exp_T = xp.exp(-1j * T * dt / hbar).astype(xp.complex64)

RENORM_INTERVAL = 200   # renormalize every this many steps to suppress drift

def psi_to_rgb_dark_bg(psi_matrix):
    amplitude = xp.abs(psi_matrix)
    phase     = xp.angle(psi_matrix)

    # Per-frame normalization: peak amplitude always maps to full brightness
    max_amp = float(xp.max(amplitude)) * 0.3 if float(xp.max(amplitude)) > 0 else 1.0

    H = (phase + xp.pi) / (2 * xp.pi)
    S = xp.ones_like(H)
    V_ch = xp.minimum((amplitude / max_amp) ** 2, 1.0)

    hsv = xp.stack([H, S, V_ch], axis=-1)
    if xp is not _np:
        hsv = hsv.get()
    rgb = hsv_to_rgb(_np.array(hsv, dtype=_np.float32))
    return (rgb * 255).astype(_np.uint8)

# Frames are converted to uint8 RGB images at storage time rather than kept as
# raw complex psi arrays — avoids holding hundreds of full-resolution complex64
# grids in memory and avoids re-running the HSV conversion on every render/export pass.
frames = [psi_to_rgb_dark_bg(psi)]

print("Evaluating time evolution steps:")
for step in range(1, num_steps + 1):
    psi *= exp_V
    psi_k = xp.fft.fft2(psi)
    psi_k *= exp_T
    psi = xp.fft.ifft2(psi_k)
    psi *= exp_V

    # FIX 4 (continued): periodic renormalization to keep norm from drifting
    if step % RENORM_INTERVAL == 0:
        psi = normalize(psi)

    if step % steps_per_frame == 0:
        frames.append(psi_to_rgb_dark_bg(psi))

    if step % 80 == 0 or step == num_steps:
        percent = (step / num_steps) * 100
        bar_length = int(step / num_steps * 40)
        progress_bar = "█" * bar_length + "-" * (40 - bar_length)
        sys.stdout.write(f"\r[{progress_bar}] {percent:.1f}% Simulation Complete")
        sys.stdout.flush()

print("\nSimulation finished successfully. Rendering animation plot...")

# 5B. EXPORT PROMPT

print("\n" + "="*60)
print("Export Options:")
print("  1. GIF (Pillow)")
print("  2. MP4 (FFmpeg)")
print("  3. No Export")
print("="*60)
export_choice = input("Select export format (1/2/3): ").strip()

# 6. HSV WAVEFUNCTION VISUALIZATION

fig, ax = plt.subplots(figsize=(8, 8))

spatial_extent_Å = [-extent/(2*Å), extent/(2*Å), -extent/(2*Å), extent/(2*Å)]
img = ax.imshow(frames[0], extent=spatial_extent_Å, origin='lower')

# Potential barrier overlays
if num_barriers >= 1:
    ax.axvspan(0, a/Å, color='#00ff00', alpha=0.5, linewidth=0.01, label='Barrier 1 (3 Å Wide)')
if num_barriers >= 2:
    ax.axvspan((a+b)/Å, (2*a+b)/Å, color='#00ff00', alpha=0.5, linewidth=0.01, label='Barrier 2 (3 Å Wide)')

ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.set_xlabel("Spatial Dimension X ($Å$)")
ax.set_ylabel("Spatial Dimension Y ($Å$)")

titles = {
    0: "Free Quantum Propagation (No Barriers)",
    1: "Quantum Tunneling (Single Barrier)",
    2: "Quantum Tunneling (Double Barrier)"
}
ax.set_title(titles[num_barriers])

if num_barriers > 0:
    leg = ax.legend(loc='upper right', framealpha=0.2, edgecolor='#333333')
    for text in leg.get_texts():
        text.set_color('#00ff00')

def animate_frame(idx):
    img.set_data(frames[idx])
    return [img]

ani = animation.FuncAnimation(
    fig,
    animate_frame,
    frames=len(frames),
    interval=16,
    blit=False
)

# EXPORT :)

if export_choice == '1':
    print("Encoding and saving animation to GIF...")
    ani.save('quantum_tunneling.gif', writer='pillow', fps=60)
    print("Export complete! Saved as 'quantum_tunneling.gif'")
elif export_choice == '2':
    print("Encoding and saving animation to MP4...")
    ani.save(
        'quantum_tunneling.mp4',
        writer='ffmpeg',
        fps=60,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
    )
    print("Export complete! Saved as 'quantum_tunneling.mp4'")
elif export_choice == '3':
    print("Skipping export.")
else:
    print("Invalid choice. Skipping export.")

plt.show()