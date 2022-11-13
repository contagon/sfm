import matplotlib.pyplot as plt
import numpy as np

def plotCoordinateFrame(T_0f, size=1, linewidth=2, k='-', ax=None):
    # https://github.com/ethz-asl/kalibr/blob/master/Schweizer-Messer/sm_python/python/sm/plotCoordinateFrame.py
    """Plot a coordinate frame on a 3d axis. In the resulting plot,
    x = red, y = green, z = blue.
    
    plotCoordinateFrame(axis, T_0f, size=1, linewidth=3)
    Arguments:
    axis: an axis of type matplotlib.axes.Axes3D
    T_0f: The 4x4 transformation matrix that takes points from the frame of interest, to the plotting frame
    size: the length of each line in the coordinate frame
    linewidth: the width of each line in the coordinate frame
    Usage is a bit irritating:
    import mpl_toolkits.mplot3d.axes3d as p3
    import pylab as pl
    f1 = pl.figure(1)
    # old syntax
    # a3d = p3.Axes3D(f1)
    # new syntax
    a3d = f1.add_subplot(111, projection='3d')
    # ... Fill in T_0f, the 4x4 transformation matrix
    plotCoordinateFrame(a3d, T_0f)
    see http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/tutorial.html for more details
    """
    # \todo fix this check.
    #if type(axis) != axes.Axes3D:
    #    raise TypeError("axis argument is the wrong type. Expected matplotlib.axes.Axes3D, got %s" % (type(axis)))
    if ax is None:
        ax = plt.gca()
    
    p_f = np.array([ [ 0,0,0,1], [size,0,0,1], [0,size,0,1], [0,0,size,1]]).T
    p_0 = np.dot(T_0f,p_f)
    # X-axis

    X = np.append( [p_0[:,0].T] , [p_0[:,1].T], axis=0 )
    Y = np.append( [p_0[:,0].T] , [p_0[:,2].T], axis=0 )
    Z = np.append( [p_0[:,0].T] , [p_0[:,3].T], axis=0 )
    ax.plot3D(X[:,0],X[:,1],X[:,2],'r'+k, linewidth=linewidth)
    ax.plot3D(Y[:,0],Y[:,1],Y[:,2],'g'+k, linewidth=linewidth)
    ax.plot3D(Z[:,0],Z[:,1],Z[:,2],'b'+k, linewidth=linewidth)
    
def set_axes_equal(ax):
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])