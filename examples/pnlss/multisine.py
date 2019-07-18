import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pyvib.forcing import multisine
from pyvib.common import db


def plot(sig):
    plt.figure()
    for i, u in enumerate(sig):
        U = np.fft.fft(u)
        plt.subplot(2,2,1+i)
        plt.plot(db(U),'+')
        plt.xlabel('Frequency line')
        plt.ylabel('Amplitude (dB)')
        plt.title(f'Phase realization {i+1}')
        plt.subplot(2,2,3+i)
        plt.plot(np.angle(U),'+')
        plt.xlabel('Frequency line')
        plt.ylabel('Phase (rad)')
        plt.title(f'Phase realization {i+1}')


# Generate two realizations of a full multisine with 1000 samples and
# excitation up to one third of the Nyquist frequency
N = 1000  # One thousand samples
kind = 'full'  # Full multisine
f2 = round(N//6)  # Excitation up to one sixth of the sample frequency
R = 2   # Two phase realizations
u, lines, freq = multisine(f2=f2,N=N,lines=kind,R=R)
# Check spectra
plot(u)
# The two realizations have the same amplitude spectrum, but different phase
# realizations (uniformly distributed between [-π,π))

# Generate a random odd multisine where the excited odd lines are split in
# groups of three consecutive lines and where one line is randomly chosen in
# each group to be a detection line (i.e. without excitation)
N = 1000
kind = 'oddrandom'
f2 = round(N//6)
R = 1
# One out of three consecutive odd lines is randomly selected to be a detection line
ngroup = 3
u1,lines, freq = multisine(f2=f2,N=N,lines=kind,R=R,ngroup=ngroup)
# Generate another phase realization of this multisine with the same excited
# lines and detection lines
u2,*_ = multisine(N=N,lines=lines,R=1)
plot((u1[0], u2[0]))

# Change the coloring and rms level of a default multisine
u3 = multisine()[0][0]
b, a = signal.cheby1(2,10,2*0.1)  # Filter coefficients
U3 = np.fft.fft(u3)
worN = 2*np.pi*np.arange(len(u3))/len(u3)
w, h = signal.freqz(b,a,worN)
U_colored = h * U3  # Filtered spectrum
u_colored = np.real(np.fft.ifft(U_colored))  # Colored multisine signal
u_scaled = 2*u_colored/np.std(u_colored)  # Scaled signal to rms value 2
# (u_colored is zero-mean)
plot((u3,u_scaled))

plt.show()
