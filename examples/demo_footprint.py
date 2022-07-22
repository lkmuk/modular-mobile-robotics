from mrobotics.models.footprint import footprint_rectangle

import matplotlib.pyplot as plt
car1 = footprint_rectangle(xf=1.2, xr=0.2, width=0.8)

fig,ax = plt.subplots()
ax.axis('equal')
ax.grid('minor') 
car1.draw_pose(0, 0, 3.14/4, ax)
car1.draw_pose(4.00, 5.0, -3.14/3, ax)  
plt.show()