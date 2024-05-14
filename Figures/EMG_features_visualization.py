import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(3,4, figsize=(20,12))

# IEMG
Y = [-2, -2, -2, -2, 2, 2, 2, 2]
X = np.linspace(0,7,8)
ax[0][0].stem(X,Y, label="x", linefmt="C0-")
ax[0][0].stem(X, np.abs(Y), label="|x|", linefmt="C1--", markerfmt="1")
ax[0][0].legend()
ax[0][0].set_title(r"$\text{MAV} = \langle | x_i |\rangle=$" + str(np.sum(np.abs(Y))/X.size))
ax[0][0].grid()

# MAV1
Y = [-2, -2, -2, -2, 2, 2, 2, 2]
W = [0.5, 0.5, 1, 1, 1, 1, 0.5, 0.5]
X = np.linspace(0,7,8)
ax[0][1].stem(X,Y, label="x", linefmt="C0-")
ax[0][1].stem(X, np.abs(Y) * W, label="w|x|", linefmt="C1--", markerfmt="1")
ax[0][1].stem(X,W, label="w", linefmt="C2--", markerfmt="o")
ax[0][1].legend()
ax[0][1].set_title(r"$\text{MAV1} = \langle w_i| x_i |\rangle=$" + str(np.sum(W*np.abs(Y))/X.size))
ax[0][1].grid()

# MAV2
Y = [-2, -2, -2, -2, 2, 2, 2, 2]
W = [0, 0.5, 1, 1, 1, 1, 0.5, 0]
X = np.linspace(0,7,8)
ax[0][2].stem(X,Y, label="x", linefmt="C0-")
ax[0][2].stem(X, np.abs(Y) * W, label="w|x|", linefmt="C1--", markerfmt="1")
ax[0][2].stem(X,W, label="w", linefmt="C2--", markerfmt="o")
ax[0][2].legend()
ax[0][2].set_title(r"$\text{MAV2} = \langle w_i| x_i | \rangle=$" + str(np.sum(W*np.abs(Y))/X.size))
ax[0][2].grid()

# Variance
Y = [-2, -2, -2, -2, 2, 2, 2, 2]
X = np.linspace(0,7,8)
ax[0][3].stem(X,Y, label="x", linefmt="C0-")
ax[0][3].stem(X, np.square(Y), label=r"$x^2$", linefmt="C1--", markerfmt="1")
ax[0][3].legend()
ax[0][3].set_title(r"$\text{VAR} = \langle (x_i-\mu)^2 \rangle=$" + str(np.var(Y)))
ax[0][3].grid()

# TM3
Y = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
X = np.linspace(0,7,8)
ax[1][0].stem(X,Y, label="x", linefmt="C0-")
ax[1][0].stem(X, Y**3, label=r"$x^3$", linefmt="C1--", markerfmt="1")
ax[1][0].legend()
ax[1][0].set_title(r"$\text{TM3} = | \langle x_i^3 \rangle |=$" + str(np.abs(np.sum(Y**3)/Y.size)))
ax[1][0].grid()

# RMS
Y = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
X = np.linspace(0,7,8)
ax[1][1].stem(X,Y, label="x", linefmt="C0-")
ax[1][1].stem(X, Y**2, label=r"$x^2$", linefmt="C1--", markerfmt="1")
ax[1][1].legend()
ax[1][1].set_title(r"$\text{RMS} =\sqrt{\langle x_i^2 \rangle}=$" + str(np.sqrt(np.mean(np.square(Y)))))
ax[1][1].grid()

# Log detector
Y = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
X = np.linspace(0,7,8)
ax[1][2].stem(X,Y, label="x", linefmt="C0-")
ax[1][2].stem(X, np.log(np.abs(Y)), label=r"$\ln(|x|)$", linefmt="C1--", markerfmt="1")
ax[1][2].legend()
ax[1][2].set_title(r"$\text{LOG} =\exp{\langle \ln{|x_i|} \rangle}=$" + str(np.exp(  (np.sum(np.log(np.abs(Y))))/Y.size  )))
ax[1][2].grid()

# Waveform length
Y = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
X = np.linspace(0,7,8)
ax[1][3].stem(X,Y, label="x", linefmt="C0-")
ax[1][3].stem(X[:-1], np.abs(np.diff(Y)), label=r"$|\Delta x|$", linefmt="C1--", markerfmt="1")
ax[1][3].legend()
ax[1][3].set_title(r"$\text{AAC} = \langle |x_{i+1}-x_{i}|  \rangle = $" + str(np.sum(np.abs(np.diff(Y)))/Y.size))
ax[1][3].grid()

# DASDV
Y = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
X = np.linspace(0,7,8)
ax[2][0].stem(X,Y, label="x", linefmt="C0-")
ax[2][0].stem(X[:-1], (np.diff(Y))**2, label=r"$(\Delta x)^2$", linefmt="C1--", markerfmt="1")
ax[2][0].legend()
ax[2][0].set_title(r"$\text{DASDV}^2 = {\langle (x_{i+1}-x_{i})^2  \rangle} =16/7$")
ax[2][0].grid()

# MYOP
Y = np.array([-2, -2, -0.5, -0.5, 0.5, 2, 2, 2])
X = np.linspace(0,7,8)
ax[2][1].stem(X,Y, label="x", linefmt="C0-")
ax[2][1].stem(X, np.abs(Y), label=r"$|x|$", linefmt="C1--", markerfmt="1")
ax[2][1].axhline(y=1, color="green", linestyle = "--", label=r"$\epsilon$")
ax[2][1].legend()
ax[2][1].set_title(r"$\text{MYOP} = \langle |x|>\epsilon \rangle =$" + str(np.count_nonzero(np.abs(Y) > 1)) + "/" + str(Y.size))
ax[2][1].grid()

# WAMP
Y = np.array([-2, -2, -0.5, -0.5, 0.25, 2, 2, 2])
X = np.linspace(0,7,8)
ax[2][2].stem(X,Y, label="x", linefmt="C0-")
ax[2][2].stem(X[:-1], np.abs(np.diff(Y)), label=r"$|\Delta x|$", linefmt="C1--", markerfmt="1")
ax[2][2].axhline(y=1, color="green", linestyle = "--", label=r"$\epsilon$")
ax[2][2].legend()
ax[2][2].set_title(r"$\text{WAMP} = \sum |x_{i+1}-x_{i}| \geq \epsilon=$" + str((   np.count_nonzero(np.abs(np.diff(Y))> 1))))
ax[2][2].grid()

# SSC
Y = np.array([-2, 0, 2, 0, -2, -2, 2, 2])
X = np.linspace(0,7,8)
ax[2][3].stem(X,Y, label="x", linefmt="C0-")
ax[2][3].stem(X, [0,0, 1, 0,0,0 , 1, 0] , label=r"SSC", linefmt="C1--", markerfmt="1")
ax[2][3].legend()
ax[2][3].set_title(r"$\text{SSC} =2$")
ax[2][3].grid()


fig.suptitle(r"Visualization of all features for $N=8$", fontsize='xx-large')
plt.tight_layout()

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
filepath = os.path.join(dir_path, "EMG_features_visualization.svg")
plt.savefig(filepath)





# # And the absolute value of the moments
# out_dict["TM3"] = np.abs(np.sum(emg**3)/N)
# out_dict["TM4"] = np.abs(np.sum(emg**4)/N)
# out_dict["TM5"] = np.abs(np.sum(emg**5)/N)

# # RMS is already in the basic_features()

# # v-order is practically the same as RMS in the optimal case (v=2)

# # Log detector
# out_dict["LOG"] = np.exp(  (np.sum(np.log(np.abs(emg))))/N  )

# # Waveform length
# out_dict["WL"] = np.sum(np.abs(dif_emg))

# # Average amplitude change is the same as the waveform length

# # Difference absolute standard deviation value
# out_dict["DASDV"] = np.sqrt(  np.sum(dif_emg**2) / (N-1)  )

# # Myopulse rate
# out_dict["MYOP"] = np.count_nonzero(np.abs(emg) > threshold) / N

# # Willison amplitude
# out_dict["WAMP"] = np.count_nonzero(np.abs(dif_emg)> threshold)

# # SLope sign change
# out_dict["SSC"] = np.count_nonzero((emg[1:-1]-emg[:-2])*(emg[1:-1]-emg[2:]) > threshold)

# # Overall muscle activity level: RMS is a measure of the amplitude of the EMG signal and reflects the overall muscle activity level
# out_dict["RMS"] = feat_gen.rms(emg)