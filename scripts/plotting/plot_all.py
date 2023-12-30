import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy

# Assuming 'song_id' is a column in both dataframes
arousal_df = pd.read_csv("data/annotations/arousal_cont_average.csv")
valence_df = pd.read_csv("data/annotations/valence_cont_average.csv")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

time = [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
        20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
        25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
        30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
        35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
        40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000]

# Loop through the dataframe rows
rs=[]
rsqs=[]
for index, row in arousal_df.iterrows():
    song_id = row['song_id']
    a = row[1:].tolist()
    
    # Assuming there is a corresponding row in valence_df with the same song_id
    v_row = valence_df[valence_df["song_id"]==song_id].iloc[0]
    v = v_row[1:].tolist()
    
    # Generate a random color
    color = np.random.rand(3,)
    
    ax.scatter(a, v, time, c=color, s=1)
    ax.plot(a, v, time, color=color)
    

    _, _, r_value1, _, _ = scipy.stats.linregress(a, time)
    _, _, r_value2, _, _ = scipy.stats.linregress(v, time)
    rs.append(abs(r_value1))
    rs.append(abs(r_value2))
    rsqs.append(r_value1**2)
    rsqs.append(r_value2**2)

ax.set_xlabel('Arousal')
ax.set_ylabel('Valence')
ax.set_zlabel('Time (ms)')

ax.set_xlim3d(-1, 1)
ax.set_zlim3d(15000, 45000)
ax.set_ylim3d(-1, 1)

circle_radius = 1  # Adjust the radius as needed
circle_center = (0, 0, 15000)

theta = np.linspace(0, 2 * np.pi, 100)
x_circle = circle_center[0] + circle_radius * np.cos(theta)
y_circle = circle_center[1] + circle_radius * np.sin(theta)
z_circle = circle_center[2] * np.ones_like(theta)

ax.plot(x_circle, y_circle, z_circle, color='k', alpha=0.5, linestyle="dashed")
ax.set_proj_type('ortho')

ax.plot((-1,1),(0,0),(15000, 15000), color='k', alpha=0.5, linestyle="dashed")
ax.plot((0,0),(-1,1),(15000, 15000), color='k', alpha=0.5, linestyle="dashed")
ax.plot((0,0),(0,0),(15000, 50000), color='k', alpha=0.5, linestyle="dashed")

plt.show()
print(f"AVG R: {sum(rs)/len(rs)}\nAVG R-SQ: {sum(rsqs)/len(rsqs)}")